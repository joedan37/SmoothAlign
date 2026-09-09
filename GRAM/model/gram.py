import os
import json
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from utils.logger import LOGGER
from .general_module import TokenMasker, MMGeneralModule, Contra_head, Match_head
from utils.distributed import all_gather_with_grad, concat_all_gather, all_gather_list
from torch.nn import LayerNorm as LayerNorm
from easydict import EasyDict as edict
from utils.volume import compute_gaussian_prior, volume_computation4,volume_computation3, volume_computation5


class GRAM(MMGeneralModule):
    """ VLP pretraining """
    def __init__(self, config):
        super().__init__()
    
        self.config = config
        self.construct_vision_encoder()
        self.construct_audio_encoder()
        self.construct_multimodal_encoder()

        contra_dim = self.config.contra_dim
        self.contra_head_t = Contra_head(self.multimodal_dim, contra_dim)
        self.contra_head_s = Contra_head(self.multimodal_dim, contra_dim)
        self.contra_head_v = Contra_head(self.vision_dim, contra_dim)
        self.contra_head_a = Contra_head(self.audio_dim, contra_dim)
        self.contra_head_d = Contra_head(self.vision_dim, contra_dim)
        self.contra_head_va = nn.Linear(self.vision_dim + self.audio_dim, contra_dim)
        self.contra_head_vs = nn.Linear(self.vision_dim + self.multimodal_dim, contra_dim)
        self.contra_head_vas = nn.Linear(self.vision_dim + self.audio_dim + self.multimodal_dim, contra_dim)
        self.gaussian_prior_enabled = getattr(config, 'gaussian_prior_enabled', False)
        self.gaussian_prior_tau = getattr(config, 'gaussian_prior_tau', 1.0)
        self.gaussian_prior_distance = str(
            getattr(config, 'gaussian_prior_distance', 'symmetric_kl')
        ).lower()
        if self.gaussian_prior_distance not in {
            'symmetric_kl', 'mean_cosine', 'wasserstein2', 'csd'
        }:
            raise ValueError(
                f'Unsupported gaussian_prior_distance: {self.gaussian_prior_distance}'
            )
        self.gaussian_prior_modalities_from_task = getattr(config, 'gaussian_prior_modalities_from_task', False)
        self.gaussian_prior_modalities_from_training = getattr(config, 'gaussian_prior_modalities_from_training', False)
        self.gaussian_prior_complete_v5_modalities = getattr(config, 'gaussian_prior_complete_v5_modalities', False)
        self.gaussian_prior_confidence_enabled = getattr(config, 'gaussian_prior_confidence_enabled', False)
        self.gaussian_prior_confidence_temp = getattr(config, 'gaussian_prior_confidence_temp', 0.07)
        self.gaussian_prior_confidence_power = getattr(config, 'gaussian_prior_confidence_power', 0.5)
        self.gaussian_prior_confidence_floor = getattr(config, 'gaussian_prior_confidence_floor', 0.0)
        self.gaussian_prior_area_softening_enabled = getattr(
            config, 'gaussian_prior_area_softening_enabled', self.gaussian_prior_enabled
        )
        self.same_video_multi_positive = getattr(config, 'same_video_multi_positive', False)
        self.same_video_v2t_positive_weight = getattr(config, 'same_video_v2t_positive_weight', 1.0)
        self.itm_prior_softening_enabled = getattr(config, 'itm_prior_softening_enabled', False)
        self.gaussian_prior_itm_softening_enabled = getattr(
            config, 'gaussian_prior_itm_softening_enabled', self.itm_prior_softening_enabled
        )
        self.itm_prior_softening_min_weight = getattr(config, 'itm_prior_softening_min_weight', 0.05)
        self.itm_hard_negative_use_prior = getattr(config, 'itm_hard_negative_use_prior', True)
        self.middle_layer_index = getattr(config, 'middle_layer_index', None)
        self.contra_temp = nn.Parameter(torch.tensor(0.07))
        self.itm_head = Match_head(self.multimodal_dim)
        self.vision_frame_embedding = nn.Parameter(0.02 * torch.randn(1, self.config.max_vision_sample_num, self.multimodal_dim))
        self.audio_frame_embedding = nn.Parameter(0.02 * torch.randn(1, self.config.max_audio_sample_num, self.multimodal_dim))
        self.hidden_trans_vision_multimodal = nn.Sequential(nn.Linear(self.vision_dim, self.multimodal_dim),LayerNorm(self.multimodal_dim, eps=1e-12))
        self.hidden_trans_audio_multimodal = nn.Sequential(nn.Linear(self.audio_dim, self.multimodal_dim),LayerNorm(self.multimodal_dim, eps=1e-12))
        self.hidden_trans_subtitle_multimodal = nn.Sequential(nn.Linear(self.multimodal_dim, self.multimodal_dim),LayerNorm(self.multimodal_dim, eps=1e-12))
        self.vision_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.audio_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.subtitle_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.beam_size  = config.beam_size
        self.itm_ratio = config.itm_ratio   
        self.max_omni_caption_len = config.max_omni_caption_len
        self.max_caption_len = config.max_caption_len
        self.max_subtitle_len = config.max_subtitle_len





   
    def construct_multimodal_encoder(self):    
        
        from .text_encoders.bert.bert import BertForMaskedLM, BertConfig
     
        bertconfig = BertConfig.from_pretrained("./weights/encoders/bert/bert-base-uncased")
        bertconfig.add_cross_attention = True
        bertconfig.is_decoder = True
        self.multimodal_encoder = BertForMaskedLM.from_pretrained("./weights/encoders/bert/bert-base-uncased",config = bertconfig )
        self.multimodal_dim = 768

        if self.config.checkpointing:
            self.multimodal_encoder._set_gradient_checkpointing(self.multimodal_encoder.bert.encoder, True)

        from transformers import BertTokenizer


        self.multimodal_encoder.tokenizer = BertTokenizer.from_pretrained('./weights/encoders/bert/bert-base-uncased')
        self.multimodal_encoder.tokenizer.bos_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.multimodal_encoder.tokenizer.eos_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        self.multimodal_encoder.tokenizer.pad_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        self.multimodal_encoder.tokenizer.mask_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

        self.text_masker = TokenMasker(mask_token = self.multimodal_encoder.tokenizer.mask_token_id, range_start=106, range_end = 30522)

        

    def batch_get(self, batch, key):
        if key in batch:
            return batch[key]


        elif key == 'caption_tokens':

            caption_tokens = self.multimodal_encoder.tokenizer(batch.raw_captions,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_caption_len,
                                                    return_tensors="pt").to(torch.device('cuda'))
         
            batch[key] = caption_tokens
        
        elif key == 'subtitle_tokens':
         
            subtitle_tokens = self.multimodal_encoder.tokenizer(batch.raw_subtitles,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_subtitle_len,
                                                    return_tensors="pt")
            subtitle_tokens = subtitle_tokens.to(torch.device('cuda'))
            batch[key] = subtitle_tokens
                                        

        elif key == 'vision_caption_tokens':
            caption_tokens = self.multimodal_encoder.tokenizer(batch.vision_captions,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_caption_len,
                                                    return_tensors="pt")

            caption_tokens = caption_tokens.to(torch.device('cuda'))
            batch[key] = caption_tokens


        
        elif key == 'audio_caption_tokens':
            caption_tokens = self.multimodal_encoder.tokenizer(batch.audio_captions,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_caption_len,
                                                    return_tensors="pt")

            caption_tokens = caption_tokens.to(torch.device('cuda'))
            batch[key] = caption_tokens

        elif key == 'omni_caption_tokens':
            caption_tokens = self.multimodal_encoder.tokenizer(batch.omni_captions,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_omni_caption_len,
                                                    return_tensors="pt")

            caption_tokens = caption_tokens.to(torch.device('cuda'))
            batch[key] = caption_tokens


        elif key == 'caption_output':
            caption_tokens = self.batch_get(batch, 'caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_output = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            batch[key] = caption_output

        elif key == 'caption_middle_output':
            caption_tokens = self.batch_get(batch, 'caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_output = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask,
                                            output_hidden_states=True)
            layer_index = self.middle_layer_index
            if layer_index is None:
                layer_index = self.multimodal_encoder.bert.config.num_hidden_layers // 2
            batch[key] = caption_output.hidden_states[layer_index]



        elif key == 'vision_caption_output':
            caption_tokens = self.batch_get(batch, 'vision_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_output = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            batch[key] = caption_output


        elif key == 'audio_caption_output':
            caption_tokens = self.batch_get(batch, 'audio_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_output = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            batch[key] = caption_output

       
        elif key == 'subtitle_output':
            subtitle_tokens = self.batch_get(batch, 'subtitle_tokens')
            input_ids = subtitle_tokens.input_ids
            attention_mask = subtitle_tokens.attention_mask
            subtitle_output = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            batch[key] = subtitle_output

        elif key == 'subtitle_middle_output':
            subtitle_tokens = self.batch_get(batch, 'subtitle_tokens')
            input_ids = subtitle_tokens.input_ids
            attention_mask = subtitle_tokens.attention_mask
            subtitle_output = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask,
                                            output_hidden_states=True)
            layer_index = self.middle_layer_index
            if layer_index is None:
                layer_index = self.multimodal_encoder.bert.config.num_hidden_layers // 2
            batch[key] = subtitle_output.hidden_states[layer_index]
  

        elif key == 'vision_output':
            vision_output = self.forward_vision_encoder(batch.vision_pixels)
            batch[key] = vision_output

        elif key == 'vision_middle_output':
            vision_output = self.forward_vision_encoder(
                batch.vision_pixels,
                return_intermediate_features=True,
                intermediate_layer=self.middle_layer_index,
            )
            if isinstance(vision_output, tuple):
                _, vision_middle_output = vision_output
            else:
                vision_middle_output = vision_output
            batch[key] = vision_middle_output
            
        elif key == 'depth_output':
            depth_output = self.forward_vision_encoder(batch.depth_pixels)
            batch[key] = depth_output
        
        elif key == 'audio_output':
            audio_output = self.forward_audio_encoder(batch.audio_spectrograms) 
            batch[key] = audio_output

        elif key == 'audio_middle_output':
            audio_output = self.forward_audio_encoder(
                batch.audio_spectrograms,
                return_intermediate_features=True,
                intermediate_layer=self.middle_layer_index,
            )
            if isinstance(audio_output, tuple):
                _, audio_middle_output = audio_output
            else:
                audio_middle_output = audio_output
            batch[key] = audio_middle_output


        elif key == 'condition_feats_v':
            vision_output = self.batch_get(batch, 'vision_output')
            condition_feats_v = self.get_multimodal_forward_input_vision(vision_output)
            batch[key] = condition_feats_v
            
        elif key == 'condition_feats_d':
            vision_output = self.batch_get(batch, 'depth_output')
            condition_feats_d = self.get_multimodal_forward_input_vision(vision_output)
            batch[key] = condition_feats_d
            
        elif key == 'condition_feats_a':
            audio_output = self.batch_get(batch, 'audio_output')
            condition_feats_a = self.get_multimodal_forward_input_audio(audio_output)
            batch[key] = condition_feats_a

        elif key == 'condition_feats_s':
            subtitle_output = self.batch_get(batch, 'subtitle_output')
            condition_feats_s = self.get_multimodal_forward_input_subtitle(subtitle_output)
            batch[key] = condition_feats_s

        elif key == 'condition_feats_va':
            condition_feats_v = self.batch_get(batch, 'condition_feats_v')
            condition_feats_a = self.batch_get(batch, 'condition_feats_a')
            condition_feats_va = torch.cat((condition_feats_v, condition_feats_a),dim=1)
            batch[key] = condition_feats_va

        elif key == 'condition_feats_vs':
            condition_feats_v = self.batch_get(batch, 'condition_feats_v')
            condition_feats_s = self.batch_get(batch, 'condition_feats_s')
            condition_feats_vs = torch.cat((condition_feats_v, condition_feats_s),dim=1)
            batch[key] = condition_feats_vs

        elif key == 'condition_feats_vas':
            condition_feats_v = self.batch_get(batch, 'condition_feats_v')
            condition_feats_a = self.batch_get(batch, 'condition_feats_a')
            condition_feats_s = self.batch_get(batch, 'condition_feats_s')
            condition_feats_vas = torch.cat((condition_feats_v, condition_feats_a, condition_feats_s),dim=1)
            batch[key] = condition_feats_vas
            
        elif key == 'condition_feats_vasd':
            condition_feats_v = self.batch_get(batch, 'condition_feats_v')
            condition_feats_a = self.batch_get(batch, 'condition_feats_a')
            condition_feats_s = self.batch_get(batch, 'condition_feats_s')
            condition_feats_d = self.batch_get(batch, 'condition_feats_d')
            condition_feats_vas = torch.cat((condition_feats_v, condition_feats_a, condition_feats_s, condition_feats_d),dim=1)
            batch[key] = condition_feats_vas



        elif key == 'feat_v':
            vision_output = self.batch_get(batch, 'vision_output')
            vision_output_pooled = self.pool_vision_for_contra(vision_output)
            feat_v = self.contra_head_v(vision_output_pooled)
            feat_v = F.normalize(feat_v,dim=-1)
            batch[key] = feat_v
        
        elif key == 'feat_d':
            depth_output = self.batch_get(batch, 'depth_output')
            depth_output_pooled = self.pool_vision_for_contra(depth_output)
            feat_d = self.contra_head_d(depth_output_pooled) #TODO do we have to use a contra head depth?
            feat_d = F.normalize(feat_d,dim=-1)
            batch[key] = feat_d
        
        elif key == 'feat_a':
            audio_output = self.batch_get(batch, 'audio_output')
            audio_output_pooled = self.pool_audio_for_contra(audio_output)
            feat_a = self.contra_head_a(audio_output_pooled)
            feat_a = F.normalize(feat_a,dim=-1)
            batch[key] = feat_a       

        elif key == 'feat_s':
            subtitle_output = self.batch_get(batch, 'subtitle_output')
            subtitle_output_pooled = self.pool_text_for_contra(subtitle_output)
            feat_s = self.contra_head_s(subtitle_output_pooled) 
            feat_s = F.normalize(feat_s,dim=-1)
            batch[key] = feat_s

        elif key == 'feat_t':
            caption_output = self.batch_get(batch, 'caption_output')
            caption_output_pooled = self.pool_text_for_contra(caption_output)
            feat_t = self.contra_head_t(caption_output_pooled) 
            feat_t = F.normalize(feat_t,dim=-1)
            batch[key] = feat_t

        elif key == 'prior_feat_t':
            caption_output = self.batch_get(batch, 'caption_middle_output')
            caption_output_pooled = self.pool_text_for_contra(caption_output)
            feat_t = self.contra_head_t(caption_output_pooled)
            feat_t = F.normalize(feat_t,dim=-1)
            batch[key] = feat_t

        elif key == 'prior_feat_v':
            vision_output = self.batch_get(batch, 'vision_middle_output')
            vision_output_pooled = self.pool_vision_for_contra(vision_output)
            feat_v = self.contra_head_v(vision_output_pooled)
            feat_v = F.normalize(feat_v,dim=-1)
            batch[key] = feat_v

        elif key == 'prior_feat_a':
            audio_output = self.batch_get(batch, 'audio_middle_output')
            audio_output_pooled = self.pool_audio_for_contra(audio_output)
            feat_a = self.contra_head_a(audio_output_pooled)
            feat_a = F.normalize(feat_a,dim=-1)
            batch[key] = feat_a

        elif key == 'prior_feat_s':
            subtitle_output = self.batch_get(batch, 'subtitle_middle_output')
            subtitle_output_pooled = self.pool_text_for_contra(subtitle_output)
            feat_s = self.contra_head_s(subtitle_output_pooled)
            feat_s = F.normalize(feat_s,dim=-1)
            batch[key] = feat_s

        elif key == 'feat_va':
            vision_output = self.batch_get(batch, 'vision_output')
            vision_output_pooled = self.pool_vision_for_contra(vision_output)
            audio_output = self.batch_get(batch, 'audio_output')
            audio_output_pooled = self.pool_audio_for_contra(audio_output)
            feat_va = torch.cat((vision_output_pooled, audio_output_pooled), dim=1)
            feat_va = self.contra_head_va(feat_va)
            feat_va = F.normalize(feat_va,dim=-1)
            batch[key] = feat_va

        elif key == 'feat_vs': 
            vision_output = self.batch_get(batch, 'vision_output')
            vision_output_pooled = self.pool_vision_for_contra(vision_output)
            subtitle_output = self.batch_get(batch, 'subtitle_output')
            subtitle_output_pooled = self.pool_text_for_contra(subtitle_output)
            feat_vs = torch.cat((vision_output_pooled, subtitle_output_pooled), dim=1)
            feat_vs = self.contra_head_vs(feat_vs)
            feat_vs = F.normalize(feat_vs,dim=-1) 
            batch[key] = feat_vs

        elif key == 'feat_vas':     
            vision_output = self.batch_get(batch, 'vision_output')
            vision_output_pooled = self.pool_vision_for_contra(vision_output)
            audio_output = self.batch_get(batch, 'audio_output')
            audio_output_pooled = self.pool_audio_for_contra(audio_output)
            subtitle_output = self.batch_get(batch, 'subtitle_output')
            subtitle_output_pooled = self.pool_text_for_contra(subtitle_output)
            feat_vas = torch.cat((vision_output_pooled, audio_output_pooled, subtitle_output_pooled), dim=1)
            feat_vas = self.contra_head_vas(feat_vas)
            feat_vas = F.normalize(feat_vas,dim=-1)
            batch[key] = feat_vas  
            
            

        elif key == 'feat_t_omni_caption':
            caption_tokens = self.batch_get(batch, 'omni_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_tokens = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            caption_tokens_pooled = self.pool_text_for_contra(caption_tokens)
            feat_t = self.contra_head_t(caption_tokens_pooled) 
            feat_t = F.normalize(feat_t,dim=-1)
            batch[key] = feat_t

        elif key == 'feat_t_vision_caption':
            caption_tokens = self.batch_get(batch, 'vision_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_tokens = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            caption_tokens_pooled = self.pool_text_for_contra(caption_tokens)
            feat_t = self.contra_head_t(caption_tokens_pooled) 
            feat_t = F.normalize(feat_t,dim=-1)
            batch[key] = feat_t

        elif key == 'feat_t_audio_caption':
            caption_tokens = self.batch_get(batch, 'audio_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_tokens = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            caption_tokens_pooled = self.pool_text_for_contra(caption_tokens)
            feat_t = self.contra_head_t(caption_tokens_pooled) 
            feat_t = F.normalize(feat_t,dim=-1)
            batch[key] = feat_t

        return batch[key] 

    def gaussian_prior_feature_keys(self, batch, task=None):
        if self.gaussian_prior_modalities_from_training:
            feature_keys = ['prior_feat_t', 'prior_feat_v', 'prior_feat_a']
            if "raw_subtitles" in batch.keys():
                feature_keys.append('prior_feat_s')
            return feature_keys

        if self.gaussian_prior_complete_v5_modalities:
            feature_keys = ['prior_feat_t', 'prior_feat_v', 'prior_feat_a']
            task_uses_subtitles = task is not None and any('s' in subtask for subtask in task.split('%')[1:])
            if task_uses_subtitles and "raw_subtitles" in batch.keys():
                feature_keys.append('prior_feat_s')
            return feature_keys

        if not self.gaussian_prior_modalities_from_task or task is None:
            return ['prior_feat_t', 'prior_feat_v', 'prior_feat_a']

        modality_codes = ['t']
        for subtask in task.split('%')[1:]:
            for code in ('v', 'a', 's'):
                if code in subtask and code not in modality_codes:
                    modality_codes.append(code)

        feature_keys = []
        for code in modality_codes:
            if code == 's' and "raw_subtitles" not in batch.keys():
                continue
            feature_keys.append({
                't': 'prior_feat_t',
                'v': 'prior_feat_v',
                'a': 'prior_feat_a',
                's': 'prior_feat_s',
            }[code])
        return feature_keys

    def gaussian_prior_for_batch(self, batch, task=None):
        feature_keys = self.gaussian_prior_feature_keys(batch, task)
        modality_features = [self.batch_get(batch, key) for key in feature_keys]
        modality_features = [concat_all_gather(feature.detach()) for feature in modality_features]
        prior = compute_gaussian_prior(
            modality_features,
            tau=self.gaussian_prior_tau,
            distance_type=self.gaussian_prior_distance,
        )
        if self.gaussian_prior_confidence_enabled:
            prior = self.apply_gaussian_prior_confidence(prior, modality_features)
        return prior

    @staticmethod
    def pool_prior_feature_for_confidence(feature):
        if feature.dim() > 2:
            feature = feature.reshape(feature.shape[0], -1, feature.shape[-1]).mean(dim=1)
        return feature.float()

    def compute_gaussian_prior_sample_confidence(self, modality_features):
        valid_features = [feature for feature in modality_features if feature is not None]
        if len(valid_features) < 2:
            return None

        pooled_features = [
            F.normalize(self.pool_prior_feature_for_confidence(feature), dim=-1)
            for feature in valid_features
        ]
        batch_size = pooled_features[0].size(0)
        confidence_temp = max(float(self.gaussian_prior_confidence_temp), 1e-6)
        pair_confidences = []

        for idx_i in range(len(pooled_features)):
            for idx_j in range(idx_i + 1, len(pooled_features)):
                logits = pooled_features[idx_i] @ pooled_features[idx_j].T
                logits = logits / confidence_temp
                row_prob = F.softmax(logits, dim=1).diag()
                col_prob = F.softmax(logits, dim=0).diag()
                confidence = torch.sqrt((row_prob * col_prob).clamp_min(0.0))
                if batch_size > 1:
                    random_prob = 1.0 / batch_size
                    confidence = ((confidence - random_prob) / (1.0 - random_prob)).clamp(0.0, 1.0)
                pair_confidences.append(confidence)

        sample_confidence = torch.stack(pair_confidences, dim=0).mean(dim=0)
        confidence_power = max(float(self.gaussian_prior_confidence_power), 1e-6)
        sample_confidence = sample_confidence.clamp(0.0, 1.0).pow(confidence_power)
        confidence_floor = min(max(float(self.gaussian_prior_confidence_floor), 0.0), 1.0)
        if confidence_floor > 0:
            sample_confidence = confidence_floor + (1.0 - confidence_floor) * sample_confidence
        return sample_confidence

    def apply_gaussian_prior_confidence(self, prior_matrix, modality_features):
        sample_confidence = self.compute_gaussian_prior_sample_confidence(modality_features)
        if sample_confidence is None:
            return prior_matrix

        pair_confidence = torch.sqrt(
            sample_confidence.unsqueeze(1) * sample_confidence.unsqueeze(0)
        ).to(device=prior_matrix.device, dtype=prior_matrix.dtype)
        gated_prior = prior_matrix * pair_confidence
        gated_prior.fill_diagonal_(1.0)
        return gated_prior

    def expand_prior_to_all_gather(self, prior, batch_size):
        rank = dist.get_rank()
        row_start = rank * batch_size
        row_end = row_start + batch_size
        return prior[row_start:row_end]

    def apply_prior_to_logits(self, logits, targets, prior_matrix, positive_mask=None):
        hard_push_mask = (1.0 - prior_matrix).clamp(min=1e-6, max=1.0).to(logits)
        weighted_logits = logits + hard_push_mask.log()
        if positive_mask is not None:
            weighted_logits = torch.where(positive_mask, logits, weighted_logits)
        else:
            weighted_logits.scatter_(1, targets.unsqueeze(1), logits.gather(1, targets.unsqueeze(1)))
        return weighted_logits

    def get_same_video_positive_mask(self, batch, targets, device):
        local_ids = list(batch.ids)
        gathered_ids = [item for worker_ids in all_gather_list(local_ids) for item in worker_ids]
        positive_mask = torch.tensor(
            [[local_id == gathered_id for gathered_id in gathered_ids] for local_id in local_ids],
            device=device,
            dtype=torch.bool,
        )
        positive_mask.scatter_(1, targets.unsqueeze(1), True)
        return positive_mask

    @staticmethod
    def multi_positive_cross_entropy(logits, positive_mask, label_smoothing=0.0):
        positive_mask = positive_mask.to(dtype=logits.dtype)
        positive_count = positive_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        target_distribution = positive_mask / positive_count
        if label_smoothing > 0:
            uniform_distribution = torch.full_like(target_distribution, 1.0 / logits.size(1))
            target_distribution = (1.0 - label_smoothing) * target_distribution + label_smoothing * uniform_distribution
        log_probs = F.log_softmax(logits, dim=1)
        return -(target_distribution * log_probs).sum(dim=1).mean()

    @staticmethod
    def weighted_multi_positive_cross_entropy(logits, positive_weights, label_smoothing=0.0):
        positive_weights = positive_weights.to(device=logits.device, dtype=logits.dtype)
        positive_sum = positive_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        target_distribution = positive_weights / positive_sum
        if label_smoothing > 0:
            uniform_distribution = torch.full_like(target_distribution, 1.0 / logits.size(1))
            target_distribution = (1.0 - label_smoothing) * target_distribution + label_smoothing * uniform_distribution
        log_probs = F.log_softmax(logits, dim=1)
        return -(target_distribution * log_probs).sum(dim=1).mean()

    def build_v2t_positive_weights(self, positive_mask, targets):
        positive_weights = positive_mask.to(dtype=torch.float32)
        if self.same_video_v2t_positive_weight == 1.0:
            return positive_weights
        diagonal_mask = torch.zeros_like(positive_mask)
        diagonal_mask.scatter_(1, targets.unsqueeze(1), True)
        off_diagonal_positive = positive_mask & ~diagonal_mask
        off_diagonal_weight = float(self.same_video_v2t_positive_weight)
        positive_weights = positive_weights.masked_fill(off_diagonal_positive, off_diagonal_weight)
        positive_weights.scatter_(1, targets.unsqueeze(1), 1.0)
        return positive_weights

    @staticmethod
    def weighted_cross_entropy(logits, targets, sample_weights):
        per_sample_loss = F.cross_entropy(logits, targets, reduction='none')
        sample_weights = sample_weights.to(device=logits.device, dtype=per_sample_loss.dtype)
        return (per_sample_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)


    def forward(self, batch, task, compute_loss=True):
        batch = edict(batch)
        ### gram-27m pretraining
        #if 'vision_captions' in batch or 'audio_captions' in batch or 'omni_captions' in batch:
        #    assert compute_loss
        #    return self.forward_vast27m(batch, task)

        ### other datasets pretraining or finetuning
        output_ls = []
        task_ls = task.split('_')


        for task in task_ls:
            if task.startswith('ret'):
                ret_dict = self.forward_ret(batch, task, compute_loss=compute_loss)
                output_ls.append(ret_dict)

            elif task.startswith('cap'):
                cap_dict = self.forward_cap(batch, task, compute_loss=compute_loss)
                output_ls.append(cap_dict)

            elif task.startswith('qa'):
                qa_dict = self.forward_qa(batch, task, compute_loss=compute_loss)
                output_ls.append(qa_dict)
            
            else:
                raise NotImplementedError
    

        output_dict = {k:v for dic in output_ls for k,v in dic.items()  }
        return output_dict


    def forward_vast27m(self, batch, task):
     
        output_ls = []
        task_ls = task.split('_')


        for task in task_ls:
            if task.startswith('ret'):
                ret_dict = self.forward_ret_vast27m(batch, task)
                output_ls.append(ret_dict)

            elif task.startswith('cap'):
                cap_dict = self.forward_cap_vast27m(batch, task)
                output_ls.append(cap_dict)
            
            else:
                raise NotImplementedError
    

        output_dict = {k:v for dic in output_ls for k,v in dic.items()  }
        return output_dict

    def compute_slice_scores(self, slice_multimodal_vision_input, slice_input_ids, slice_attention_mask):
            
        slice_output = self.multimodal_encoder.bert(input_ids = slice_input_ids,
                                                    attention_mask = slice_attention_mask,
                                                    encoder_hidden_states=slice_multimodal_vision_input).last_hidden_state
        slice_scores = F.softmax(self.itm_head(slice_output[:,0]),dim=1)[:,1]

        return slice_scores


    def forward_ret(self, batch, task, compute_loss=True):
        
        if isinstance(batch.raw_captions[0],list): #### test
            batch.raw_captions = [i for j in batch.raw_captions for i in j]
        subtasks = task.split('%')[1:]  
        if compute_loss:
            loss_dict={}
            loss_itc = []
            loss_itm = []
            loss_area = []

            #Extract text features
            feat_t = self.batch_get(batch,'feat_t')
            feat_t_all = concat_all_gather(feat_t)
            #Extract visual features
            feat_v = self.batch_get(batch,'feat_v')
            feat_v_all = concat_all_gather(feat_v)
            #Extract audio features
            feat_a = self.batch_get(batch,'feat_a')
            feat_a_all = concat_all_gather(feat_a)
            #Extract subtitles features
            if "raw_subtitles" in batch.keys():
                feat_s = self.batch_get(batch,'feat_s')
                feat_s_all = concat_all_gather(feat_s)
            #extract depth features
            if "depth_pixels" in batch.keys():
                feat_d = self.batch_get(batch,'feat_d')
                feat_d_all = concat_all_gather(feat_d)
            #TODO extract other k features
            
            caption_tokens = self.batch_get(batch, 'caption_tokens')
            input_ids, attention_mask = caption_tokens.input_ids, caption_tokens.attention_mask
            input_ids_collate = concat_all_gather(input_ids)
            attention_mask_collate = concat_all_gather(attention_mask)


            #       VOLUME LOSS COMPUTATION
            #           VOLUME ITC

            #Volume (Text, batch_all)
            if "raw_subtitles" in batch.keys():
                if "depth_pixels" in batch.keys():
                    volume = volume_computation5(feat_t,feat_v_all,feat_a_all,feat_s_all,feat_d_all)
                    volume = volume / self.contra_temp
                else:
                    volume = volume_computation4(feat_t,feat_v_all,feat_a_all,feat_s_all)
                    volume = volume / self.contra_temp
            else:
                volume = volume_computation3(feat_t,feat_v_all,feat_a_all)
                volume = volume / self.contra_temp
            #AreaT (Video,batch_all)
            if "raw_subtitles" in batch.keys():
                if "depth_pixels" in batch.keys():
                    volumeT = volume_computation5(feat_t_all,feat_v,feat_a,feat_s,feat_d).T
                    volumeT = volumeT / self.contra_temp
                else:
                    volumeT = volume_computation4(feat_t_all,feat_v,feat_a,feat_s).T
                    volumeT = volumeT / self.contra_temp
            else:
                volumeT = volume_computation3(feat_t_all,feat_v,feat_a).T
                volumeT = volumeT / self.contra_temp
            rank = dist.get_rank()
            bs = feat_t.size(0)
            targets = rank * bs + torch.arange(bs, device=volume.device, dtype=torch.long)
            positive_mask = self.get_same_video_positive_mask(batch, targets, volume.device) if self.same_video_multi_positive else None

            logits_t2d = -volume
            logits_d2t = -volumeT
            logits_t2d_for_itm_sampling = logits_t2d
            logits_d2t_for_itm_sampling = logits_d2t
            prior_matrix = None
            if (
                    self.gaussian_prior_enabled
                    or self.gaussian_prior_area_softening_enabled
                    or self.gaussian_prior_itm_softening_enabled
            ):
                prior_matrix = self.expand_prior_to_all_gather(self.gaussian_prior_for_batch(batch, task), bs)
                if self.gaussian_prior_area_softening_enabled:
                    logits_t2d = self.apply_prior_to_logits(logits_t2d, targets, prior_matrix, positive_mask)
                    logits_d2t = self.apply_prior_to_logits(logits_d2t, targets, prior_matrix, positive_mask)
                    if self.itm_hard_negative_use_prior:
                        logits_t2d_for_itm_sampling = logits_t2d
                        logits_d2t_for_itm_sampling = logits_d2t

            if positive_mask is not None:
                v2t_positive_weights = self.build_v2t_positive_weights(positive_mask, targets)
                loss = (
                        self.multi_positive_cross_entropy(logits_t2d, positive_mask, label_smoothing=0.1)
                        + self.weighted_multi_positive_cross_entropy(logits_d2t, v2t_positive_weights, label_smoothing=0.1)
                ) / 2
            else:
                loss = (
                        F.cross_entropy(logits_t2d, targets, label_smoothing=0.1) #d2a
                        + F.cross_entropy(logits_d2t, targets, label_smoothing=0.1) #a2d
                ) / 2

            loss_area.append(loss)


            #   AREA VID ITM 
            #vid_sim = feat_t @ feat_v_all.T
            #vid_simT = feat_v @ feat_t_all.T
                     
            condition_feats = self.batch_get(batch, f'condition_feats_va')#self.batch_get(batch, f'condition_feats_v')
            condition_feats_collate = all_gather_with_grad(condition_feats)
            with torch.no_grad():
                weights_t2cond = F.softmax(logits_t2d_for_itm_sampling, dim=1) + 1e-4
                weights_cond2t = F.softmax(logits_d2t_for_itm_sampling, dim=1) + 1e-4
                if positive_mask is not None:
                    weights_t2cond.masked_fill_(positive_mask, 0)
                    weights_cond2t.masked_fill_(positive_mask, 0)
                else:
                    weights_t2cond[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
                    weights_cond2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

            condition_feats_neg = []
            neg_indices_t2cond = []
            for b in range(bs): 
                neg_idx = torch.multinomial(weights_t2cond[b], 1).item()
                neg_indices_t2cond.append(neg_idx)
                condition_feats_neg.append(condition_feats_collate[neg_idx])
            condition_feats_neg = torch.stack(condition_feats_neg, dim=0)
            neg_indices_t2cond = torch.tensor(neg_indices_t2cond, device=volume.device, dtype=torch.long)

            text_ids_neg = []
            text_atts_neg = []
            neg_indices_cond2t = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_cond2t[b], 1).item()
                neg_indices_cond2t.append(neg_idx)
                text_ids_neg.append(input_ids_collate[neg_idx])
                text_atts_neg.append(attention_mask_collate[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)
            neg_indices_cond2t = torch.tensor(neg_indices_cond2t, device=volume.device, dtype=torch.long)
        
            input_ids_1 = torch.cat((input_ids, input_ids, text_ids_neg),dim=0)
            attention_mask_1 = torch.cat((attention_mask, attention_mask, text_atts_neg),dim=0)
            
            condition_feats = torch.cat((condition_feats,condition_feats_neg,condition_feats),dim=0)
            output = self.multimodal_encoder.bert(input_ids = input_ids_1,
                                        attention_mask = attention_mask_1,
                                        encoder_hidden_states=condition_feats).last_hidden_state
            batch_size = condition_feats_neg.shape[0]
            logits = self.itm_head(output[:,0].half())
            ground_truth = torch.zeros(batch_size*3).long().cuda()
            ground_truth[:batch_size] = 1
            if self.gaussian_prior_itm_softening_enabled and prior_matrix is not None:
                neg_prior_t2cond = prior_matrix.gather(1, neg_indices_t2cond.unsqueeze(1)).squeeze(1)
                neg_prior_cond2t = prior_matrix.gather(1, neg_indices_cond2t.unsqueeze(1)).squeeze(1)
                neg_weight_t2cond = (1.0 - neg_prior_t2cond).clamp(
                    min=self.itm_prior_softening_min_weight,
                    max=1.0,
                )
                neg_weight_cond2t = (1.0 - neg_prior_cond2t).clamp(
                    min=self.itm_prior_softening_min_weight,
                    max=1.0,
                )
                itm_sample_weights = torch.cat(
                    (
                        torch.ones(batch_size, device=volume.device, dtype=prior_matrix.dtype),
                        neg_weight_t2cond,
                        neg_weight_cond2t,
                    ),
                    dim=0,
                )
                loss = self.weighted_cross_entropy(logits, ground_truth, itm_sample_weights) #itm (dtm)
            else:
                loss = F.cross_entropy(logits,ground_truth) #itm (dtm)
            loss_itm.append(self.itm_ratio * loss)

            

            for task in subtasks:


                # #### compute_itc
                # assert task in ['tv','ta','tva','tvs','tvas']
                # #feat_cond = self.batch_get(batch,f'feat_{task[1:]}')
                # #feat_cond_all = concat_all_gather(feat_cond)
                # sim_cond2t = torch.matmul(feat_cond, feat_t_all.permute(1,0))
                # sim_cond2t = sim_cond2t / self.contra_temp
                # sim_t2cond = torch.matmul(feat_t, feat_cond_all.permute(1,0))
                # sim_t2cond = sim_t2cond / self.contra_temp  # [batch_size, batch_size*num_gpu]
                # rank = dist.get_rank()
                # bs = feat_t.size(0)
                # targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(feat_cond.device)
                # loss = (
                #     F.cross_entropy(sim_cond2t, targets, label_smoothing=0.1)
                #     + F.cross_entropy(sim_t2cond, targets, label_smoothing=0.1)
                # ) / 2
                
                loss_itc.append(torch.tensor(0))#*loss)

                #### compute_itm
         
                # condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
                # condition_feats_collate = all_gather_with_grad(condition_feats)
                # with torch.no_grad():
                #     weights_t2cond = F.softmax(sim_t2cond, dim=1) + 1e-4
                #     weights_t2cond[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
                #     weights_cond2t = F.softmax(sim_cond2t, dim=1) + 1e-4
                #     weights_cond2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

                # condition_feats_neg = []
                # for b in range(bs): 
                #     neg_idx = torch.multinomial(weights_t2cond[b], 1).item()
                #     condition_feats_neg.append(condition_feats_collate[neg_idx])
                # condition_feats_neg = torch.stack(condition_feats_neg, dim=0)

                # text_ids_neg = []
                # text_atts_neg = []
                # for b in range(bs):
                #     neg_idx = torch.multinomial(weights_cond2t[b], 1).item()
                #     text_ids_neg.append(input_ids_collate[neg_idx])
                #     text_atts_neg.append(attention_mask_collate[neg_idx])

                # text_ids_neg = torch.stack(text_ids_neg, dim=0)
                # text_atts_neg = torch.stack(text_atts_neg, dim=0)
        
                # input_ids_1 = torch.cat((input_ids, input_ids, text_ids_neg),dim=0)
                # attention_mask_1 = torch.cat((attention_mask, attention_mask, text_atts_neg),dim=0)
            
                # condition_feats = torch.cat((condition_feats,condition_feats_neg,condition_feats),dim=0)
                # output = self.multimodal_encoder.bert(input_ids = input_ids_1,
                #                             attention_mask = attention_mask_1,
                #                             encoder_hidden_states=condition_feats).last_hidden_state
                # batch_size = condition_feats_neg.shape[0]
                # logits = self.itm_head(output[:,0].half())
                # ground_truth = torch.zeros(batch_size*3).long().cuda()
                # ground_truth[:batch_size] = 1
                # loss = F.cross_entropy(logits,ground_truth)
                #loss_itm.append(torch.tensor(0))#*(self.itm_ratio * loss))

            loss_itc = sum(loss_itc)/len(loss_itc)
            loss_dict['loss_itc'] = loss_itc          
            loss_itm = sum(loss_itm)/len(loss_itm)
            loss_dict['loss_itm'] = loss_itm
            loss_area = sum(loss_area)/len(loss_area)
            loss_dict['loss_area'] = loss_area
            return loss_dict
          
        else:

            evaluation_dict = {}
            feat_t = self.batch_get(batch,'feat_t')
            evaluation_dict['feat_t'] = feat_t 
            feat_v = self.batch_get(batch,'feat_v')
            evaluation_dict['feat_v'] = feat_v 
            feat_a = self.batch_get(batch,'feat_a')
            evaluation_dict['feat_a'] = feat_a
            
            if "raw_subtitles" in batch.keys(): 
                feat_s = self.batch_get(batch,'feat_s')
                evaluation_dict['feat_s'] = feat_s
            if "depth_pixels" in batch.keys():
                feat_d = self.batch_get(batch,'feat_d')
                evaluation_dict['feat_d'] = feat_d
            

            caption_tokens = self.batch_get(batch,'caption_tokens')
            evaluation_dict['input_ids'] = caption_tokens.input_ids
            evaluation_dict['attention_mask'] = caption_tokens.attention_mask
            for task in subtasks:
                #### compute_itc
                assert task in ['tv','ta','tva','tvs','tvas','tvasd']
                # feat_cond = self.batch_get(batch,f'feat_{task[1:]}')
                # evaluation_dict[f'feat_cond_{task}'] = feat_cond

                condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
                evaluation_dict[f'condition_feats_{task}'] = condition_feats

            return evaluation_dict

    def forward_cap(self, batch, task, compute_loss=True):
        subtasks = task.split('%')[1:]

        if compute_loss:

            loss_dict = {}
            loss_ls_cap = []

            caption_tokens = self.batch_get(batch, 'caption_tokens')
            input_ids, attention_mask = caption_tokens.input_ids, caption_tokens.attention_mask
            input_ids, txt_labels = self.text_masker(input_ids, 0.6)
            
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1).clone()
            attention_mask[:, : seq_len, : seq_len] = torch.tril(attention_mask[:, : seq_len, : seq_len])

            for task in subtasks:
                assert task in ['tv','ta','tva','tvs','tvas']
                condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
                output = self.multimodal_encoder(input_ids = input_ids,
                                                attention_mask = attention_mask,
                                                encoder_hidden_states=condition_feats,
                                                labels = txt_labels)
                loss_ls_cap.append(output.loss)

            loss_cap = sum(loss_ls_cap)/len(loss_ls_cap)
            loss_dict['loss_cap'] = loss_cap
            return loss_dict

        else:
            evaluation_dict = {}
            for task in subtasks:
                assert task in ['tv','ta','tva','tvs','tvas']
                condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')

                batch_size = condition_feats.shape[0]
                if self.config.captioner_mode:
                    batch_size *=self.config.generate_nums

                init_input_ids = torch.ones(batch_size, 1).long().cuda().fill_(self.multimodal_encoder.tokenizer.bos_token_id)
                init_attention_mask = init_input_ids.new_ones(batch_size, 1, 1)
                
                if self.config.captioner_mode:
                    condition_feats = condition_feats.unsqueeze(1).expand(-1, self.config.generate_nums,-1,-1).reshape(-1,*condition_feats.shape[1:])
                    outputs = self.multimodal_encoder.generate( input_ids=init_input_ids,
                                                            attention_mask=init_attention_mask,
                                                            do_sample = True,
                                                            top_k = 10,
                                                            encoder_hidden_states=condition_feats,
                                                            max_new_tokens=self.max_caption_len,
                                                            eos_token_id=self.multimodal_encoder.tokenizer.sep_token_id,
                                                            pad_token_id=self.multimodal_encoder.tokenizer.pad_token_id) 

                                                        
                else:
                    outputs = self.multimodal_encoder.generate( input_ids=init_input_ids,
                                        attention_mask=init_attention_mask,
                                        encoder_hidden_states=condition_feats,
                                        max_new_tokens=self.max_caption_len,
                                        num_beams=self.beam_size,
                                        eos_token_id=self.multimodal_encoder.tokenizer.sep_token_id,
                                        pad_token_id=self.multimodal_encoder.tokenizer.pad_token_id,
                                        length_penalty=0.6) 
                                                        
                outputs_newgen = outputs[:,1:]
                captions = self.multimodal_encoder.tokenizer.batch_decode(outputs_newgen, skip_special_tokens=True)
                evaluation_dict[f'generated_captions_{task}'] = captions

            return evaluation_dict



    def forward_qa(self, batch, task, compute_loss=True):
        subtasks = task.split('%')[1:]
        raw_questions = batch.raw_questions
        raw_answers = batch.raw_answers
      
        if isinstance(raw_questions[0],list): #### test
            # raw_batch_size = len(raw_questions)
            num_questions = [len(i) for i in raw_questions]
            raw_questions = [j for d in raw_questions for j in d]

        question_tokens = self.multimodal_encoder.tokenizer(raw_questions,
                                                            padding="max_length",
                                                            truncation=True,
                                                            max_length=self.max_caption_len,
                                                            return_tensors="pt").to(torch.device('cuda'))

        question_tokens_ids, question_tokens_mask = question_tokens.input_ids, question_tokens.attention_mask

        if compute_loss:

            loss_dict = {}
            loss_ls_qa = []

            answer_tokens = self.multimodal_encoder.tokenizer(raw_answers,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=10,
                                                    return_tensors="pt")
    
            answer_tokens = answer_tokens.to(torch.device('cuda'))
            answer_tokens_ids, answer_tokens_mask = answer_tokens.input_ids, answer_tokens.attention_mask
            input_ids, txt_labels = self.text_masker(answer_tokens_ids, 0.99)
            input_ids = torch.cat((question_tokens_ids,input_ids),dim=1)
            attention_mask = torch.cat((question_tokens_mask,answer_tokens_mask),dim=1)
            dummy_labels = (-100*torch.ones_like(question_tokens_ids)).cuda()
            txt_labels = torch.cat((dummy_labels,txt_labels),dim=1)

            #### part-causal attention mask
            question_len = question_tokens_ids.shape[1]
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1).clone()
            attention_mask[:, question_len: seq_len, question_len: seq_len] = torch.tril(attention_mask[:, question_len: seq_len, question_len: seq_len])
            attention_mask[:, :question_len, question_len:seq_len] = 0


            for task in subtasks:
                assert task in ['tv','ta','tva','tvs','tvas']
                condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
                output = self.multimodal_encoder(input_ids = input_ids,
                                attention_mask = attention_mask,
                                encoder_hidden_states=condition_feats,
                                labels = txt_labels)
                loss_ls_qa.append(output.loss)
            
            loss_qa = sum(loss_ls_qa)/len(loss_ls_qa)
            loss_dict['loss_qa'] = loss_qa
            return loss_dict
        
        else:
            evaluation_dict = {} 
            batch_size = sum(num_question)
            init_input_ids = torch.ones(batch_size, 1).long().cuda().fill_(self.multimodal_encoder.tokenizer.bos_token_id)
            init_input_ids = torch.cat((question_tokens['input_ids'],init_input_ids),dim=1)
            question_len = question_tokens['input_ids'].shape[1]
            seq_len = init_input_ids.shape[1]
            attention_mask = question_tokens['attention_mask'].unsqueeze(1).expand(-1, question_len, -1).clone()
            init_attention_mask = self.multimodal_encoder.update_attention_mask(attention_mask)


            for task in subtasks:
                assert task in ['tv','ta','tva','tvs','tvas']
                condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
            
                condition_feats_expand = []
                for i in range(condition_feats.shape[0]):
                    condition_feats_expand.append( condition_feats[i:i+1].expand(num_questions[i],-1,-1))
                condition_feats = torch.cat(condition_feats_expand,dim=0)
                batch_size = condition_feats.shape[0]
                       
                outputs = self.multimodal_encoder.generate( input_ids=init_input_ids,
                                                            attention_mask=init_attention_mask,
                                                            encoder_hidden_states=condition_feats,
                                                            max_new_tokens=10,
                                                            num_beams=self.beam_size,
                                                            eos_token_id=self.multimodal_encoder.tokenizer.sep_token_id,
                                                            pad_token_id=self.multimodal_encoder.tokenizer.pad_token_id) 
                
                outputs_newgen = outputs[:,seq_len:]
                answers = self.multimodal_encoder.tokenizer.batch_decode(outputs_newgen, skip_special_tokens=True)
                print(answers)
                evaluation_dict[f'generated_answers_{task}'] = answers


            return evaluation_dict




    def forward_cap_vast27m(self, batch, task):


        subtasks = task.split('%')[1:]
        loss_dict = {}
        loss_ls_cap = []

        for task in subtasks:
            assert task in ['tv','ta','tva','tvs','tvas']
            if task == 'tv':
                caption_tokens = self.batch_get(batch, 'vision_caption_tokens')

            elif task == 'ta':
                caption_tokens = self.batch_get(batch, 'audio_caption_tokens')
                
            else:
                caption_tokens = self.batch_get(batch, 'omni_caption_tokens')

            input_ids, attention_mask = caption_tokens.input_ids, caption_tokens.attention_mask
            input_ids, txt_labels = self.text_masker(input_ids, 0.6)
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1).clone()
            attention_mask[:, : seq_len, : seq_len] = torch.tril(attention_mask[:, : seq_len, : seq_len])
            condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
            output = self.multimodal_encoder(input_ids = input_ids,
                                            attention_mask = attention_mask,
                                            encoder_hidden_states=condition_feats,
                                            labels = txt_labels)
            loss_ls_cap.append(output.loss)

        loss_cap = sum(loss_ls_cap)/len(loss_ls_cap)
        loss_dict['loss_cap'] = loss_cap
        return loss_dict


    def forward_ret_vast27m(self, batch, task):
        

        subtasks = task.split('%')[1:]  
        if compute_loss:
            loss_dict={}
            loss_itc = []
            loss_itm = []
 
            for task in subtasks:
                #### compute_itc
                assert task in ['tv','ta','tva','tvs','tvas']
                if task == 'tv':
                    feat_t = self.batch_get(batch,'feat_t_vision_caption')
                    caption_tokens = self.batch_get(batch, 'vision_caption_tokens')
                elif task == 'ta':
                    feat_t = self.batch_get(batch,'feat_t_audio_caption')
                    caption_tokens = self.batch_get(batch, 'audio_caption_tokens')
                else:
                    feat_t = self.batch_get(batch,'feat_t_omni_caption')
                    caption_tokens = self.batch_get(batch, 'omni_caption_tokens', txt_len = self.max_omni_caption_len)
               
                feat_t_all = concat_all_gather(feat_t) 

                input_ids, attention_mask = caption_tokens.input_ids, caption_tokens.attention_mask
                input_ids_collate = concat_all_gather(input_ids)
                attention_mask_collate = concat_all_gather(attention_mask)

                feat_cond = self.batch_get(batch,f'feat_{task[1:]}')
                feat_cond_all = concat_all_gather(feat_cond)
                sim_cond2t = torch.matmul(feat_cond, feat_t_all.permute(1,0))
                sim_cond2t = sim_cond2t / self.contra_temp
                sim_t2cond = torch.matmul(feat_t, feat_cond_all.permute(1,0))
                sim_t2cond = sim_t2cond / self.contra_temp  # [batch_size, batch_size*num_gpu]
                rank = dist.get_rank()
                bs = feat_t.size(0)
                targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(feat_cond.device)
                loss = (
                    F.cross_entropy(sim_cond2t, targets, label_smoothing=0.1)
                    + F.cross_entropy(sim_t2cond, targets, label_smoothing=0.1)
                ) / 2
                
                loss_itc.append(loss)

                #### compute_itm
         
                condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
                condition_feats_collate = all_gather_with_grad(condition_feats)
                with torch.no_grad():
                    weights_t2cond = F.softmax(sim_t2cond, dim=1) + 1e-4
                    weights_t2cond[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
                    weights_cond2t = F.softmax(sim_cond2t, dim=1) + 1e-4
                    weights_cond2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

                condition_feats_neg = []
                for b in range(bs): 
                    neg_idx = torch.multinomial(weights_t2cond[b], 1).item()
                    condition_feats_neg.append(condition_feats_collate[neg_idx])
                condition_feats_neg = torch.stack(condition_feats_neg, dim=0)

                text_ids_neg = []
                text_atts_neg = []
                for b in range(bs):
                    neg_idx = torch.multinomial(weights_cond2t[b], 1).item()
                    text_ids_neg.append(input_ids_collate[neg_idx])
                    text_atts_neg.append(attention_mask_collate[neg_idx])

                text_ids_neg = torch.stack(text_ids_neg, dim=0)
                text_atts_neg = torch.stack(text_atts_neg, dim=0)
        
                input_ids_1 = torch.cat((input_ids, input_ids, text_ids_neg),dim=0)
                attention_mask_1 = torch.cat((attention_mask, attention_mask, text_atts_neg),dim=0)
            
                condition_feats = torch.cat((condition_feats,condition_feats_neg,condition_feats),dim=0)
                output = self.multimodal_encoder.bert(input_ids = input_ids_1,
                                            attention_mask = attention_mask_1,
                                            encoder_hidden_states=condition_feats).last_hidden_state
                batch_size = condition_feats_neg.shape[0]
                logits = self.itm_head(output[:,0].half())
                ground_truth = torch.zeros(batch_size*3).long().cuda()
                ground_truth[:batch_size] = 1
                loss = F.cross_entropy(logits,ground_truth)
                loss_itm.append(self.itm_ratio * loss)

            loss_itc = sum(loss_itc)/len(loss_itc)
            loss_dict['loss_itc'] = loss_itc          
            loss_itm = sum(loss_itm)/len(loss_itm)
            loss_dict['loss_itm'] = loss_itm

            return loss_dict
          
