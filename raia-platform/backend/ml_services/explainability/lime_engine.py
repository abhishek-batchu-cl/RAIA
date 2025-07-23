# LIME Explainability Engine - Local Interpretable Model-Agnostic Explanations
import lime
import lime.lime_tabular
import lime.lime_text
import lime.lime_image
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable
import json
from datetime import datetime
import logging
from PIL import Image
from sklearn.pipeline import Pipeline
from ..exceptions import ExplainabilityError, ValidationError

logger = logging.getLogger(__name__)

class LIMEExplainabilityEngine:
    """LIME (Local Interpretable Model-Agnostic Explanations) engine for model interpretability"""
    
    def __init__(self):
        self.explainer_cache = {}
        
    def create_tabular_explainer(self,
                               training_data: pd.DataFrame,
                               feature_names: List[str] = None,
                               categorical_features: List[int] = None,
                               categorical_names: Dict[int, List[str]] = None,
                               mode: str = 'classification') -> lime.lime_tabular.LimeTabularExplainer:
        """Create LIME explainer for tabular data"""
        
        try:
            if isinstance(training_data, pd.DataFrame):
                training_array = training_data.values
                feature_names = feature_names or list(training_data.columns)
            else:
                training_array = training_data
                
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=training_array,
                feature_names=feature_names,
                categorical_features=categorical_features,
                categorical_names=categorical_names,
                mode=mode,
                discretize_continuous=True,
                random_state=42
            )
            
            return explainer
            
        except Exception as e:
            raise ExplainabilityError(f"Failed to create LIME tabular explainer: {str(e)}")
    
    def create_text_explainer(self,
                            mode: str = 'classification',
                            bow: bool = True,
                            char_level: bool = False) -> lime.lime_text.LimeTextExplainer:
        """Create LIME explainer for text data"""
        
        try:
            explainer = lime.lime_text.LimeTextExplainer(
                mode=mode,
                bow=bow,
                char_level=char_level,
                random_state=42
            )
            
            return explainer
            
        except Exception as e:
            raise ExplainabilityError(f"Failed to create LIME text explainer: {str(e)}")
    
    def create_image_explainer(self,
                             mode: str = 'classification',
                             feature_selection: str = 'auto') -> lime.lime_image.LimeImageExplainer:
        """Create LIME explainer for image data"""
        
        try:
            explainer = lime.lime_image.LimeImageExplainer(
                mode=mode,
                feature_selection=feature_selection,
                random_state=42
            )
            
            return explainer
            
        except Exception as e:
            raise ExplainabilityError(f"Failed to create LIME image explainer: {str(e)}")
    
    def explain_tabular_instance(self,
                                model_id: str,
                                model: Any,
                                instance: Union[pd.DataFrame, np.ndarray],
                                training_data: pd.DataFrame,
                                feature_names: List[str] = None,
                                categorical_features: List[int] = None,
                                categorical_names: Dict[int, List[str]] = None,
                                num_features: int = 10,
                                num_samples: int = 5000,
                                top_labels: int = None) -> Dict[str, Any]:
        """Generate LIME explanation for a tabular instance"""
        
        try:
            # Create or get cached explainer
            cache_key = f"{model_id}_tabular"
            if cache_key not in self.explainer_cache:
                # Determine mode based on model
                mode = 'classification' if hasattr(model, 'predict_proba') else 'regression'
                
                self.explainer_cache[cache_key] = self.create_tabular_explainer(
                    training_data=training_data,
                    feature_names=feature_names,
                    categorical_features=categorical_features,
                    categorical_names=categorical_names,
                    mode=mode
                )
            
            explainer = self.explainer_cache[cache_key]
            
            # Convert instance to appropriate format
            if isinstance(instance, pd.DataFrame):
                instance_array = instance.values.flatten()
                feature_names = feature_names or list(instance.columns)
            else:
                instance_array = instance.flatten() if instance.ndim > 1 else instance
            
            # Create prediction function
            if hasattr(model, 'predict_proba'):
                predict_fn = model.predict_proba
                mode = 'classification'
            else:
                predict_fn = model.predict
                mode = 'regression'
            
            # Generate explanation
            explanation = explainer.explain_instance(
                data_row=instance_array,
                predict_fn=predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                top_labels=top_labels
            )
            
            # Format explanation for frontend
            formatted_explanation = self._format_tabular_explanation(
                explanation, model_id, instance_array, feature_names, mode
            )
            
            return formatted_explanation
            
        except Exception as e:
            raise ExplainabilityError(f"Failed to explain tabular instance: {str(e)}")
    
    def explain_text_instance(self,
                            model_id: str,
                            model: Any,
                            text: str,
                            num_features: int = 10,
                            num_samples: int = 5000,
                            top_labels: int = None) -> Dict[str, Any]:
        """Generate LIME explanation for a text instance"""
        
        try:
            # Create or get cached explainer
            cache_key = f"{model_id}_text"
            if cache_key not in self.explainer_cache:
                mode = 'classification' if hasattr(model, 'predict_proba') else 'regression'
                self.explainer_cache[cache_key] = self.create_text_explainer(mode=mode)
            
            explainer = self.explainer_cache[cache_key]
            
            # Create prediction function
            if hasattr(model, 'predict_proba'):
                predict_fn = self._create_text_predict_proba_fn(model)
                mode = 'classification'
            else:
                predict_fn = self._create_text_predict_fn(model)
                mode = 'regression'
            
            # Generate explanation
            explanation = explainer.explain_instance(
                text_instance=text,
                classifier=predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                top_labels=top_labels
            )
            
            # Format explanation for frontend
            formatted_explanation = self._format_text_explanation(
                explanation, model_id, text, mode
            )
            
            return formatted_explanation
            
        except Exception as e:
            raise ExplainabilityError(f"Failed to explain text instance: {str(e)}")
    
    def explain_image_instance(self,
                             model_id: str,
                             model: Any,
                             image: np.ndarray,
                             num_features: int = 100,
                             num_samples: int = 1000,
                             top_labels: int = None,
                             hide_color: int = 0) -> Dict[str, Any]:
        """Generate LIME explanation for an image instance"""
        
        try:
            # Create or get cached explainer
            cache_key = f"{model_id}_image"
            if cache_key not in self.explainer_cache:
                mode = 'classification' if hasattr(model, 'predict_proba') else 'regression'
                self.explainer_cache[cache_key] = self.create_image_explainer(mode=mode)
            
            explainer = self.explainer_cache[cache_key]
            
            # Create prediction function
            if hasattr(model, 'predict_proba'):
                predict_fn = model.predict_proba
                mode = 'classification'
            else:
                predict_fn = model.predict
                mode = 'regression'
            
            # Generate explanation
            explanation = explainer.explain_instance(
                image=image,
                classifier_fn=predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                top_labels=top_labels,
                hide_color=hide_color,
                random_seed=42
            )
            
            # Format explanation for frontend
            formatted_explanation = self._format_image_explanation(
                explanation, model_id, image, mode
            )
            
            return formatted_explanation
            
        except Exception as e:
            raise ExplainabilityError(f"Failed to explain image instance: {str(e)}")
    
    def _format_tabular_explanation(self,
                                  explanation: lime.explanation.Explanation,
                                  model_id: str,
                                  instance: np.ndarray,
                                  feature_names: List[str],
                                  mode: str) -> Dict[str, Any]:
        """Format tabular LIME explanation for frontend"""
        
        # Get available labels
        available_labels = explanation.available_labels()
        
        formatted_result = {
            'model_id': model_id,
            'explanation_type': 'lime_tabular',
            'mode': mode,
            'instance_data': instance.tolist(),
            'feature_names': feature_names,
            'explanations': {},
            'metadata': {
                'num_features_used': len(explanation.as_list()),
                'available_labels': available_labels,
                'intercept': explanation.intercept if hasattr(explanation, 'intercept') else None,
                'prediction_local': explanation.local_pred if hasattr(explanation, 'local_pred') else None,
                'score': explanation.score if hasattr(explanation, 'score') else None
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Format explanations for each label
        for label in available_labels:
            explanation_list = explanation.as_list(label=label)
            explanation_map = explanation.as_map()[label] if label in explanation.as_map() else []
            
            formatted_features = []
            for feature_desc, importance in explanation_list:
                # Parse feature description to get feature name and value
                if '=' in feature_desc:
                    feature_name, feature_condition = feature_desc.split('=', 1)
                    feature_name = feature_name.strip()
                    feature_condition = feature_condition.strip()
                else:
                    feature_name = feature_desc
                    feature_condition = None
                
                # Try to find the actual feature value
                feature_value = None
                if feature_names and feature_name in feature_names:
                    feature_idx = feature_names.index(feature_name)
                    if feature_idx < len(instance):
                        feature_value = instance[feature_idx]
                
                formatted_features.append({
                    'feature_name': feature_name,
                    'feature_description': feature_desc,
                    'feature_condition': feature_condition,
                    'feature_value': float(feature_value) if feature_value is not None else None,
                    'lime_importance': importance,
                    'abs_importance': abs(importance),
                    'contribution_type': 'positive' if importance > 0 else 'negative'
                })
            
            # Sort by absolute importance
            formatted_features.sort(key=lambda x: x['abs_importance'], reverse=True)
            
            formatted_result['explanations'][f'label_{label}'] = {
                'feature_importances': formatted_features,
                'top_5_features': formatted_features[:5],
                'total_positive_contribution': sum([f['lime_importance'] for f in formatted_features if f['lime_importance'] > 0]),
                'total_negative_contribution': sum([f['lime_importance'] for f in formatted_features if f['lime_importance'] < 0]),
                'explanation_strength': sum([f['abs_importance'] for f in formatted_features])
            }
        
        return formatted_result
    
    def _format_text_explanation(self,
                               explanation: lime.explanation.Explanation,
                               model_id: str,
                               text: str,
                               mode: str) -> Dict[str, Any]:
        """Format text LIME explanation for frontend"""
        
        available_labels = explanation.available_labels()
        
        formatted_result = {
            'model_id': model_id,
            'explanation_type': 'lime_text',
            'mode': mode,
            'original_text': text,
            'text_length': len(text.split()),
            'explanations': {},
            'metadata': {
                'available_labels': available_labels,
                'intercept': explanation.intercept if hasattr(explanation, 'intercept') else None,
                'prediction_local': explanation.local_pred if hasattr(explanation, 'local_pred') else None,
                'score': explanation.score if hasattr(explanation, 'score') else None
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Format explanations for each label
        for label in available_labels:
            explanation_list = explanation.as_list(label=label)
            
            # Get word-level explanations
            word_explanations = []
            text_words = text.split()
            
            for word, importance in explanation_list:
                # Find word position in original text
                word_positions = []
                for i, text_word in enumerate(text_words):
                    if text_word == word:
                        word_positions.append(i)
                
                word_explanations.append({
                    'word': word,
                    'importance': importance,
                    'abs_importance': abs(importance),
                    'contribution_type': 'positive' if importance > 0 else 'negative',
                    'positions': word_positions
                })
            
            # Sort by absolute importance
            word_explanations.sort(key=lambda x: x['abs_importance'], reverse=True)
            
            # Create highlighted text
            highlighted_text = self._create_highlighted_text(text, word_explanations)
            
            formatted_result['explanations'][f'label_{label}'] = {
                'word_importances': word_explanations,
                'top_10_words': word_explanations[:10],
                'highlighted_text': highlighted_text,
                'total_positive_contribution': sum([w['importance'] for w in word_explanations if w['importance'] > 0]),
                'total_negative_contribution': sum([w['importance'] for w in word_explanations if w['importance'] < 0]),
                'explanation_coverage': len(word_explanations) / len(text_words) if text_words else 0
            }
        
        return formatted_result
    
    def _format_image_explanation(self,
                                explanation: lime.explanation.Explanation,
                                model_id: str,
                                image: np.ndarray,
                                mode: str) -> Dict[str, Any]:
        """Format image LIME explanation for frontend"""
        
        available_labels = explanation.available_labels()
        
        formatted_result = {
            'model_id': model_id,
            'explanation_type': 'lime_image',
            'mode': mode,
            'image_shape': list(image.shape),
            'explanations': {},
            'metadata': {
                'available_labels': available_labels,
                'num_superpixels': len(explanation.segments) if hasattr(explanation, 'segments') else None,
                'intercept': explanation.intercept if hasattr(explanation, 'intercept') else None,
                'prediction_local': explanation.local_pred if hasattr(explanation, 'local_pred') else None
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Format explanations for each label
        for label in available_labels:
            # Get superpixel explanations
            explanation_list = explanation.as_list(label=label)
            
            superpixel_explanations = []
            for superpixel_id, importance in explanation_list:
                superpixel_explanations.append({
                    'superpixel_id': int(superpixel_id),
                    'importance': importance,
                    'abs_importance': abs(importance),
                    'contribution_type': 'positive' if importance > 0 else 'negative'
                })
            
            # Sort by absolute importance
            superpixel_explanations.sort(key=lambda x: x['abs_importance'], reverse=True)
            
            # Generate explanation mask and highlighted regions
            explanation_masks = self._generate_explanation_masks(explanation, label, image.shape)
            
            formatted_result['explanations'][f'label_{label}'] = {
                'superpixel_importances': superpixel_explanations,
                'top_10_superpixels': superpixel_explanations[:10],
                'explanation_masks': explanation_masks,
                'total_positive_contribution': sum([s['importance'] for s in superpixel_explanations if s['importance'] > 0]),
                'total_negative_contribution': sum([s['importance'] for s in superpixel_explanations if s['importance'] < 0]),
                'num_important_regions': len([s for s in superpixel_explanations if abs(s['importance']) > 0.01])
            }
        
        return formatted_result
    
    def _create_text_predict_proba_fn(self, model):
        """Create prediction function for text models with probability output"""
        def predict_proba_wrapper(texts):
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(texts)
            else:
                # Handle pipeline models
                if isinstance(model, Pipeline):
                    return model.predict_proba(texts)
                else:
                    raise AttributeError("Model does not support predict_proba")
        return predict_proba_wrapper
    
    def _create_text_predict_fn(self, model):
        """Create prediction function for text models with single output"""
        def predict_wrapper(texts):
            predictions = model.predict(texts)
            # Ensure 2D output for LIME
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(-1, 1)
            return predictions
        return predict_wrapper
    
    def _create_highlighted_text(self, text: str, word_explanations: List[Dict]) -> Dict[str, Any]:
        """Create highlighted text with importance scores"""
        
        words = text.split()
        highlighted_words = []
        
        # Create word importance map
        word_importance_map = {we['word']: we for we in word_explanations}
        
        for word in words:
            if word in word_importance_map:
                importance_data = word_importance_map[word]
                highlighted_words.append({
                    'word': word,
                    'importance': importance_data['importance'],
                    'highlight_type': importance_data['contribution_type'],
                    'highlight_intensity': min(abs(importance_data['importance']), 1.0)  # Normalize to 0-1
                })
            else:
                highlighted_words.append({
                    'word': word,
                    'importance': 0,
                    'highlight_type': 'neutral',
                    'highlight_intensity': 0
                })
        
        return {
            'highlighted_words': highlighted_words,
            'highlighted_html': self._generate_highlighted_html(highlighted_words)
        }
    
    def _generate_highlighted_html(self, highlighted_words: List[Dict]) -> str:
        """Generate HTML with highlighted words"""
        
        html_parts = []
        for word_data in highlighted_words:
            word = word_data['word']
            importance = word_data['importance']
            highlight_type = word_data['highlight_type']
            intensity = word_data['highlight_intensity']
            
            if highlight_type == 'positive':
                alpha = intensity * 0.7  # Scale opacity
                style = f"background-color: rgba(0, 255, 0, {alpha}); padding: 2px;"
            elif highlight_type == 'negative':
                alpha = intensity * 0.7
                style = f"background-color: rgba(255, 0, 0, {alpha}); padding: 2px;"
            else:
                style = ""
            
            if style:
                html_parts.append(f'<span style="{style}" title="Importance: {importance:.3f}">{word}</span>')
            else:
                html_parts.append(word)
        
        return ' '.join(html_parts)
    
    def _generate_explanation_masks(self, 
                                  explanation: lime.explanation.Explanation,
                                  label: int,
                                  image_shape: tuple) -> Dict[str, Any]:
        """Generate explanation masks for image visualization"""
        
        try:
            # Get positive and negative masks
            temp, mask = explanation.get_image_and_mask(
                label=label, 
                positive_only=True, 
                num_features=10, 
                hide_rest=False
            )
            positive_mask = mask.tolist()
            
            temp, mask = explanation.get_image_and_mask(
                label=label, 
                positive_only=False, 
                negative_only=True, 
                num_features=10, 
                hide_rest=False
            )
            negative_mask = mask.tolist()
            
            # Get combined mask
            temp, mask = explanation.get_image_and_mask(
                label=label, 
                positive_only=False, 
                num_features=20, 
                hide_rest=False
            )
            combined_mask = mask.tolist()
            
            return {
                'positive_regions_mask': positive_mask,
                'negative_regions_mask': negative_mask,
                'combined_mask': combined_mask,
                'mask_shape': list(mask.shape)
            }
            
        except Exception as e:
            logger.warning(f"Could not generate explanation masks: {str(e)}")
            return {
                'positive_regions_mask': None,
                'negative_regions_mask': None,
                'combined_mask': None,
                'mask_shape': list(image_shape[:2])
            }
    
    def generate_neighborhood_data(self,
                                 model_id: str,
                                 explanation: lime.explanation.Explanation) -> Dict[str, Any]:
        """Generate data about the local neighborhood used by LIME"""
        
        neighborhood_data = {
            'model_id': model_id,
            'neighborhood_size': getattr(explanation, 'mode', 'unknown'),
            'local_model_score': explanation.score if hasattr(explanation, 'score') else None,
            'intercept': explanation.intercept if hasattr(explanation, 'intercept') else None,
            'local_prediction': explanation.local_pred if hasattr(explanation, 'local_pred') else None,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Add local model coefficients if available
        if hasattr(explanation, 'local_exp'):
            neighborhood_data['local_model_coefficients'] = explanation.local_exp
        
        return neighborhood_data
    
    def compare_lime_shap_explanations(self,
                                     lime_explanation: Dict[str, Any],
                                     shap_explanation: Dict[str, Any]) -> Dict[str, Any]:
        """Compare LIME and SHAP explanations for the same instance"""
        
        comparison_result = {
            'model_id': lime_explanation['model_id'],
            'comparison_timestamp': datetime.utcnow().isoformat(),
            'feature_comparison': [],
            'agreement_metrics': {}
        }
        
        # Extract feature importance from both explanations
        lime_features = {}
        if 'explanations' in lime_explanation:
            for label_key, label_data in lime_explanation['explanations'].items():
                if 'feature_importances' in label_data:
                    for feature in label_data['feature_importances']:
                        lime_features[feature['feature_name']] = feature['lime_importance']
        
        shap_features = {}
        if 'shap_values' in shap_explanation:
            shap_values = shap_explanation['shap_values']
            if 'instance_0' in shap_values:
                for feature in shap_values['instance_0']['feature_contributions']:
                    shap_features[feature['feature_name']] = feature['shap_value']
        
        # Compare features present in both explanations
        common_features = set(lime_features.keys()) & set(shap_features.keys())
        
        for feature_name in common_features:
            lime_importance = lime_features[feature_name]
            shap_importance = shap_features[feature_name]
            
            # Calculate agreement
            sign_agreement = (lime_importance > 0) == (shap_importance > 0)
            magnitude_ratio = abs(shap_importance) / abs(lime_importance) if lime_importance != 0 else float('inf')
            
            comparison_result['feature_comparison'].append({
                'feature_name': feature_name,
                'lime_importance': lime_importance,
                'shap_importance': shap_importance,
                'sign_agreement': sign_agreement,
                'magnitude_ratio': magnitude_ratio,
                'importance_difference': abs(lime_importance - shap_importance),
                'relative_difference': abs((lime_importance - shap_importance) / max(abs(lime_importance), abs(shap_importance))) if max(abs(lime_importance), abs(shap_importance)) > 0 else 0
            })
        
        # Calculate overall agreement metrics
        if comparison_result['feature_comparison']:
            sign_agreements = [f['sign_agreement'] for f in comparison_result['feature_comparison']]
            magnitude_ratios = [f['magnitude_ratio'] for f in comparison_result['feature_comparison'] if f['magnitude_ratio'] != float('inf')]
            relative_differences = [f['relative_difference'] for f in comparison_result['feature_comparison']]
            
            comparison_result['agreement_metrics'] = {
                'sign_agreement_rate': sum(sign_agreements) / len(sign_agreements),
                'average_magnitude_ratio': np.mean(magnitude_ratios) if magnitude_ratios else None,
                'average_relative_difference': np.mean(relative_differences),
                'num_common_features': len(common_features),
                'total_lime_features': len(lime_features),
                'total_shap_features': len(shap_features)
            }
        
        return comparison_result
    
    def clear_cache(self, model_id: str = None):
        """Clear explainer cache"""
        if model_id:
            # Clear specific model cache
            keys_to_remove = [key for key in self.explainer_cache.keys() if key.startswith(model_id)]
            for key in keys_to_remove:
                del self.explainer_cache[key]
        else:
            # Clear all cache
            self.explainer_cache.clear()
        
        logger.info(f"Cleared LIME explainer cache for model_id: {model_id}")