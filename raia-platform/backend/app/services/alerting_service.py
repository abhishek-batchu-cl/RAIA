"""
Enterprise Alerting and Notification Service
Comprehensive alerting system for ML models with multiple notification channels
"""

import asyncio
import json
import logging
import smtplib
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
import uuid
from collections import defaultdict, deque
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

import pandas as pd
import numpy as np
import aiohttp
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_database

logger = logging.getLogger(__name__)
settings = get_settings()

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertStatus(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

class NotificationChannel(Enum):
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    DISCORD = "discord"

class AlertingService:
    """
    Enterprise-grade alerting and notification service
    """
    
    def __init__(self):
        self.active_alerts = {}
        self.alert_history = defaultdict(list)
        self.notification_channels = {}
        self.alert_rules = defaultdict(list)
        self.escalation_policies = {}
        self.notification_templates = self._initialize_templates()
        self.suppression_rules = []
        self.alert_dependencies = {}
        self.maintenance_windows = []
        
    def _initialize_templates(self) -> Dict[str, Dict[str, str]]:
        """
        Initialize notification templates
        """
        return {
            'performance_degradation': {
                'subject': '🚨 Model Performance Alert: {model_name}',
                'email_body': '''
                <h2>Model Performance Degradation Alert</h2>
                <p><strong>Model:</strong> {model_name} ({model_id})</p>
                <p><strong>Severity:</strong> {severity}</p>
                <p><strong>Metric:</strong> {metric_name}</p>
                <p><strong>Current Value:</strong> {current_value}</p>
                <p><strong>Threshold:</strong> {threshold}</p>
                <p><strong>Description:</strong> {description}</p>
                <p><strong>Time:</strong> {timestamp}</p>
                
                <h3>Immediate Actions Required:</h3>
                <ul>
                    {recommendations}
                </ul>
                
                <p><a href="{dashboard_url}">View Dashboard</a> | <a href="{alert_url}">Acknowledge Alert</a></p>
                ''',
                'slack_text': '🚨 *{model_name}* performance degradation detected\n*{metric_name}*: {current_value} (threshold: {threshold})\n*Severity*: {severity}\n<{dashboard_url}|View Dashboard>'
            },
            'data_drift': {
                'subject': '📊 Data Drift Alert: {model_name}',
                'email_body': '''
                <h2>Data Drift Detection Alert</h2>
                <p><strong>Model:</strong> {model_name} ({model_id})</p>
                <p><strong>Severity:</strong> {severity}</p>
                <p><strong>Drifted Features:</strong> {drifted_features}</p>
                <p><strong>Drift Score:</strong> {drift_score}</p>
                <p><strong>Description:</strong> {description}</p>
                <p><strong>Time:</strong> {timestamp}</p>
                
                <h3>Recommended Actions:</h3>
                <ul>
                    <li>Review data pipeline for changes</li>
                    <li>Consider model retraining</li>
                    <li>Monitor prediction quality closely</li>
                </ul>
                
                <p><a href="{dashboard_url}">View Analysis</a></p>
                ''',
                'slack_text': '📊 *{model_name}* data drift detected\n*Features*: {drifted_features}\n*Drift Score*: {drift_score}\n*Severity*: {severity}'
            },
            'system_error': {
                'subject': '❌ System Error Alert: {model_name}',
                'email_body': '''
                <h2>System Error Alert</h2>
                <p><strong>Model:</strong> {model_name} ({model_id})</p>
                <p><strong>Severity:</strong> {severity}</p>
                <p><strong>Error Type:</strong> {error_type}</p>
                <p><strong>Error Message:</strong> {error_message}</p>
                <p><strong>Time:</strong> {timestamp}</p>
                
                <h3>Technical Details:</h3>
                <pre>{technical_details}</pre>
                
                <p><strong>Immediate Action Required:</strong> {action_required}</p>
                ''',
                'slack_text': '❌ *{model_name}* system error\n*Error*: {error_type}\n*Message*: {error_message}\n*Severity*: {severity}'
            },
            'fairness_violation': {
                'subject': '⚖️ Fairness Violation Alert: {model_name}',
                'email_body': '''
                <h2>Model Fairness Violation Alert</h2>
                <p><strong>Model:</strong> {model_name} ({model_id})</p>
                <p><strong>Severity:</strong> {severity}</p>
                <p><strong>Violation Type:</strong> {violation_type}</p>
                <p><strong>Affected Groups:</strong> {affected_groups}</p>
                <p><strong>Metric Value:</strong> {metric_value}</p>
                <p><strong>Threshold:</strong> {threshold}</p>
                <p><strong>Time:</strong> {timestamp}</p>
                
                <h3>Compliance Actions Required:</h3>
                <ul>
                    <li>Review model predictions for affected groups</li>
                    <li>Implement bias mitigation techniques</li>
                    <li>Document remediation steps</li>
                </ul>
                ''',
                'slack_text': '⚖️ *{model_name}* fairness violation\n*Type*: {violation_type}\n*Groups*: {affected_groups}\n*Severity*: {severity}'
            }
        }
    
    async def create_alert(
        self,
        alert_type: str,
        model_id: str,
        severity: AlertSeverity,
        title: str,
        description: str,
        metadata: Dict[str, Any],
        source: str = "system",
        tags: Optional[List[str]] = None,
        related_alerts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a new alert
        
        Args:
            alert_type: Type of alert (performance_degradation, data_drift, etc.)
            model_id: Model identifier
            severity: Alert severity level
            title: Alert title
            description: Detailed description
            metadata: Additional metadata
            source: Source of the alert
            tags: Optional tags for categorization
            related_alerts: Related alert IDs
        
        Returns:
            Alert creation result
        """
        try:
            alert_id = str(uuid.uuid4())
            
            alert = {
                'alert_id': alert_id,
                'alert_type': alert_type,
                'model_id': model_id,
                'severity': severity.value,
                'status': AlertStatus.ACTIVE.value,
                'title': title,
                'description': description,
                'metadata': metadata,
                'source': source,
                'tags': tags or [],
                'related_alerts': related_alerts or [],
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow(),
                'acknowledged_at': None,
                'acknowledged_by': None,
                'resolved_at': None,
                'resolved_by': None,
                'notification_count': 0,
                'escalation_level': 0,
                'suppressed': False
            }
            
            # Check for duplicate alerts
            duplicate_alert = await self._check_duplicate_alert(alert)
            if duplicate_alert:
                return {
                    'status': 'duplicate',
                    'alert_id': duplicate_alert['alert_id'],
                    'message': 'Similar alert already active',
                    'existing_alert': duplicate_alert
                }
            
            # Check suppression rules
            if await self._is_alert_suppressed(alert):
                alert['suppressed'] = True
                alert['status'] = AlertStatus.SUPPRESSED.value
            
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history[model_id].append(alert)
            
            # Send notifications if not suppressed
            if not alert['suppressed']:
                notification_result = await self._send_notifications(alert)
                alert['notification_count'] = len(notification_result.get('sent_notifications', []))
            
            return {
                'status': 'success',
                'alert_id': alert_id,
                'alert': alert,
                'notifications_sent': not alert['suppressed']
            }
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Acknowledge an alert
        
        Args:
            alert_id: Alert identifier
            acknowledged_by: User who acknowledged the alert
            notes: Optional acknowledgment notes
        
        Returns:
            Acknowledgment result
        """
        try:
            if alert_id not in self.active_alerts:
                raise ValueError(f"Alert {alert_id} not found")
            
            alert = self.active_alerts[alert_id]
            
            if alert['status'] == AlertStatus.ACKNOWLEDGED.value:
                return {
                    'status': 'already_acknowledged',
                    'message': f"Alert already acknowledged by {alert['acknowledged_by']}",
                    'alert_id': alert_id
                }
            
            alert['status'] = AlertStatus.ACKNOWLEDGED.value
            alert['acknowledged_at'] = datetime.utcnow()
            alert['acknowledged_by'] = acknowledged_by
            alert['updated_at'] = datetime.utcnow()
            
            if notes:
                if 'acknowledgment_notes' not in alert:
                    alert['acknowledgment_notes'] = []
                alert['acknowledgment_notes'].append({
                    'note': notes,
                    'timestamp': datetime.utcnow(),
                    'user': acknowledged_by
                })
            
            # Send acknowledgment notification
            await self._send_acknowledgment_notification(alert, acknowledged_by, notes)
            
            return {
                'status': 'success',
                'alert_id': alert_id,
                'acknowledged_by': acknowledged_by,
                'acknowledged_at': alert['acknowledged_at'].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'alert_id': alert_id
            }
    
    async def resolve_alert(
        self,
        alert_id: str,
        resolved_by: str,
        resolution_notes: Optional[str] = None,
        resolution_actions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Resolve an alert
        
        Args:
            alert_id: Alert identifier
            resolved_by: User who resolved the alert
            resolution_notes: Resolution notes
            resolution_actions: Actions taken to resolve
        
        Returns:
            Resolution result
        """
        try:
            if alert_id not in self.active_alerts:
                raise ValueError(f"Alert {alert_id} not found")
            
            alert = self.active_alerts[alert_id]
            
            alert['status'] = AlertStatus.RESOLVED.value
            alert['resolved_at'] = datetime.utcnow()
            alert['resolved_by'] = resolved_by
            alert['updated_at'] = datetime.utcnow()
            
            if resolution_notes:
                alert['resolution_notes'] = resolution_notes
            
            if resolution_actions:
                alert['resolution_actions'] = resolution_actions
            
            # Calculate resolution time
            resolution_time = alert['resolved_at'] - alert['created_at']
            alert['resolution_time_seconds'] = resolution_time.total_seconds()
            
            # Send resolution notification
            await self._send_resolution_notification(alert, resolved_by, resolution_notes)
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            return {
                'status': 'success',
                'alert_id': alert_id,
                'resolved_by': resolved_by,
                'resolved_at': alert['resolved_at'].isoformat(),
                'resolution_time_minutes': resolution_time.total_seconds() / 60
            }
            
        except Exception as e:
            logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'alert_id': alert_id
            }
    
    async def configure_notification_channel(
        self,
        channel_id: str,
        channel_type: NotificationChannel,
        config: Dict[str, Any],
        enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Configure a notification channel
        
        Args:
            channel_id: Unique channel identifier
            channel_type: Type of notification channel
            config: Channel-specific configuration
            enabled: Whether channel is enabled
        
        Returns:
            Configuration result
        """
        try:
            # Validate configuration based on channel type
            validation_result = await self._validate_channel_config(channel_type, config)
            if not validation_result['valid']:
                raise ValueError(f"Invalid configuration: {validation_result['message']}")
            
            channel = {
                'channel_id': channel_id,
                'channel_type': channel_type.value,
                'config': config,
                'enabled': enabled,
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow(),
                'success_count': 0,
                'failure_count': 0,
                'last_notification': None,
                'last_success': None,
                'last_failure': None
            }
            
            # Test the channel configuration
            test_result = await self._test_notification_channel(channel)
            channel['test_result'] = test_result
            
            self.notification_channels[channel_id] = channel
            
            return {
                'status': 'success',
                'channel_id': channel_id,
                'channel_type': channel_type.value,
                'test_result': test_result,
                'enabled': enabled
            }
            
        except Exception as e:
            logger.error(f"Failed to configure notification channel: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'channel_id': channel_id
            }
    
    async def _send_notifications(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send notifications for an alert
        """
        try:
            sent_notifications = []
            failed_notifications = []
            
            # Get notification channels for this alert
            target_channels = await self._get_target_channels(alert)
            
            # Send to each channel
            for channel_id in target_channels:
                if channel_id not in self.notification_channels:
                    continue
                
                channel = self.notification_channels[channel_id]
                if not channel['enabled']:
                    continue
                
                try:
                    result = await self._send_to_channel(alert, channel)
                    if result['success']:
                        sent_notifications.append(channel_id)
                        channel['success_count'] += 1
                        channel['last_success'] = datetime.utcnow()
                    else:
                        failed_notifications.append({
                            'channel_id': channel_id,
                            'error': result.get('error', 'Unknown error')
                        })
                        channel['failure_count'] += 1
                        channel['last_failure'] = datetime.utcnow()
                    
                    channel['last_notification'] = datetime.utcnow()
                    
                except Exception as e:
                    failed_notifications.append({
                        'channel_id': channel_id,
                        'error': str(e)
                    })
                    channel['failure_count'] += 1
                    channel['last_failure'] = datetime.utcnow()
            
            return {
                'sent_notifications': sent_notifications,
                'failed_notifications': failed_notifications,
                'total_attempts': len(target_channels)
            }
            
        except Exception as e:
            logger.error(f"Error sending notifications: {e}")
            return {
                'sent_notifications': [],
                'failed_notifications': [],
                'error': str(e)
            }
    
    async def _send_to_channel(self, alert: Dict[str, Any], channel: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send notification to a specific channel
        """
        try:
            channel_type = NotificationChannel(channel['channel_type'])
            
            if channel_type == NotificationChannel.EMAIL:
                return await self._send_email_notification(alert, channel)
            elif channel_type == NotificationChannel.SLACK:
                return await self._send_slack_notification(alert, channel)
            elif channel_type == NotificationChannel.WEBHOOK:
                return await self._send_webhook_notification(alert, channel)
            elif channel_type == NotificationChannel.TEAMS:
                return await self._send_teams_notification(alert, channel)
            elif channel_type == NotificationChannel.DISCORD:
                return await self._send_discord_notification(alert, channel)
            else:
                return {
                    'success': False,
                    'error': f'Notification channel type {channel_type.value} not implemented'
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _send_email_notification(self, alert: Dict[str, Any], channel: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send email notification
        """
        try:
            config = channel['config']
            
            # Get template
            template = self.notification_templates.get(alert['alert_type'], self.notification_templates['system_error'])
            
            # Format message
            subject = template['subject'].format(**alert['metadata'], **alert)
            body = template['email_body'].format(**alert['metadata'], **alert)
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = config['from_email']
            msg['To'] = ', '.join(config['to_emails'])
            
            # Add HTML body
            html_part = MIMEText(body, 'html')
            msg.attach(html_part)
            
            # Send email
            smtp_config = config.get('smtp', {})
            with smtplib.SMTP(smtp_config.get('server', 'localhost'), smtp_config.get('port', 587)) as server:
                if smtp_config.get('use_tls', True):
                    server.starttls()
                if smtp_config.get('username') and smtp_config.get('password'):
                    server.login(smtp_config['username'], smtp_config['password'])
                
                server.send_message(msg)
            
            return {
                'success': True,
                'sent_to': config['to_emails'],
                'subject': subject
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _send_slack_notification(self, alert: Dict[str, Any], channel: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send Slack notification
        """
        try:
            config = channel['config']
            webhook_url = config['webhook_url']
            
            # Get template
            template = self.notification_templates.get(alert['alert_type'], self.notification_templates['system_error'])
            
            # Format message
            text = template['slack_text'].format(**alert['metadata'], **alert)
            
            # Create Slack payload
            payload = {
                'text': text,
                'username': config.get('username', 'RAIA Alert Bot'),
                'icon_emoji': self._get_severity_emoji(alert['severity']),
                'channel': config.get('channel', '#alerts'),
                'attachments': [{
                    'color': self._get_severity_color(alert['severity']),
                    'fields': [
                        {
                            'title': 'Model',
                            'value': alert['model_id'],
                            'short': True
                        },
                        {
                            'title': 'Severity',
                            'value': alert['severity'].upper(),
                            'short': True
                        },
                        {
                            'title': 'Alert Type',
                            'value': alert['alert_type'].replace('_', ' ').title(),
                            'short': True
                        },
                        {
                            'title': 'Time',
                            'value': alert['created_at'].strftime('%Y-%m-%d %H:%M:%S UTC'),
                            'short': True
                        }
                    ],
                    'footer': 'RAIA Platform',
                    'ts': int(alert['created_at'].timestamp())
                }]
            }
            
            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        return {
                            'success': True,
                            'channel': config.get('channel', '#alerts')
                        }
                    else:
                        return {
                            'success': False,
                            'error': f'Slack API returned status {response.status}'
                        }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _send_webhook_notification(self, alert: Dict[str, Any], channel: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send webhook notification
        """
        try:
            config = channel['config']
            webhook_url = config['url']
            
            # Create webhook payload
            payload = {
                'alert_id': alert['alert_id'],
                'alert_type': alert['alert_type'],
                'model_id': alert['model_id'],
                'severity': alert['severity'],
                'title': alert['title'],
                'description': alert['description'],
                'metadata': alert['metadata'],
                'created_at': alert['created_at'].isoformat(),
                'tags': alert['tags']
            }
            
            # Add custom headers if specified
            headers = config.get('headers', {})
            headers['Content-Type'] = 'application/json'
            
            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, headers=headers) as response:
                    if 200 <= response.status < 300:
                        return {
                            'success': True,
                            'status_code': response.status
                        }
                    else:
                        return {
                            'success': False,
                            'error': f'Webhook returned status {response.status}'
                        }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _send_teams_notification(self, alert: Dict[str, Any], channel: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send Microsoft Teams notification
        """
        try:
            config = channel['config']
            webhook_url = config['webhook_url']
            
            # Create Teams adaptive card
            payload = {
                '@type': 'MessageCard',
                '@context': 'http://schema.org/extensions',
                'themeColor': self._get_severity_color(alert['severity']),
                'summary': f"RAIA Alert: {alert['title']}",
                'sections': [{
                    'activityTitle': f"🚨 {alert['title']}",
                    'activitySubtitle': alert['description'],
                    'activityImage': config.get('icon_url'),
                    'facts': [
                        {
                            'name': 'Model ID',
                            'value': alert['model_id']
                        },
                        {
                            'name': 'Severity',
                            'value': alert['severity'].upper()
                        },
                        {
                            'name': 'Alert Type',
                            'value': alert['alert_type'].replace('_', ' ').title()
                        },
                        {
                            'name': 'Time',
                            'value': alert['created_at'].strftime('%Y-%m-%d %H:%M:%S UTC')
                        }
                    ],
                    'markdown': True
                }],
                'potentialAction': [{
                    '@type': 'OpenUri',
                    'name': 'View Dashboard',
                    'targets': [{
                        'os': 'default',
                        'uri': alert['metadata'].get('dashboard_url', '#')
                    }]
                }]
            }
            
            # Send to Teams
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        return {
                            'success': True
                        }
                    else:
                        return {
                            'success': False,
                            'error': f'Teams API returned status {response.status}'
                        }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _send_discord_notification(self, alert: Dict[str, Any], channel: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send Discord notification
        """
        try:
            config = channel['config']
            webhook_url = config['webhook_url']
            
            # Create Discord embed
            payload = {
                'username': config.get('username', 'RAIA Alert Bot'),
                'embeds': [{
                    'title': f"🚨 {alert['title']}",
                    'description': alert['description'],
                    'color': int(self._get_severity_color(alert['severity']).replace('#', ''), 16),
                    'fields': [
                        {
                            'name': 'Model ID',
                            'value': alert['model_id'],
                            'inline': True
                        },
                        {
                            'name': 'Severity',
                            'value': alert['severity'].upper(),
                            'inline': True
                        },
                        {
                            'name': 'Alert Type',
                            'value': alert['alert_type'].replace('_', ' ').title(),
                            'inline': True
                        }
                    ],
                    'timestamp': alert['created_at'].isoformat(),
                    'footer': {
                        'text': 'RAIA Platform'
                    }
                }]
            }
            
            # Send to Discord
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 204:  # Discord returns 204 for success
                        return {
                            'success': True
                        }
                    else:
                        return {
                            'success': False,
                            'error': f'Discord API returned status {response.status}'
                        }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_severity_emoji(self, severity: str) -> str:
        """
        Get emoji for severity level
        """
        emoji_map = {
            'info': ':information_source:',
            'warning': ':warning:',
            'error': ':x:',
            'critical': ':rotating_light:'
        }
        return emoji_map.get(severity, ':question:')
    
    def _get_severity_color(self, severity: str) -> str:
        """
        Get color code for severity level
        """
        color_map = {
            'info': '#36a64f',      # Green
            'warning': '#ff9500',   # Orange
            'error': '#e01e5a',     # Red
            'critical': '#8b0000'   # Dark red
        }
        return color_map.get(severity, '#808080')
    
    async def _validate_channel_config(self, channel_type: NotificationChannel, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate notification channel configuration
        """
        try:
            if channel_type == NotificationChannel.EMAIL:
                required_fields = ['from_email', 'to_emails']
                for field in required_fields:
                    if field not in config:
                        return {'valid': False, 'message': f'Missing required field: {field}'}
                
                if not isinstance(config['to_emails'], list) or not config['to_emails']:
                    return {'valid': False, 'message': 'to_emails must be a non-empty list'}
            
            elif channel_type == NotificationChannel.SLACK:
                if 'webhook_url' not in config:
                    return {'valid': False, 'message': 'Missing webhook_url for Slack channel'}
            
            elif channel_type == NotificationChannel.WEBHOOK:
                if 'url' not in config:
                    return {'valid': False, 'message': 'Missing url for webhook channel'}
            
            elif channel_type == NotificationChannel.TEAMS:
                if 'webhook_url' not in config:
                    return {'valid': False, 'message': 'Missing webhook_url for Teams channel'}
            
            elif channel_type == NotificationChannel.DISCORD:
                if 'webhook_url' not in config:
                    return {'valid': False, 'message': 'Missing webhook_url for Discord channel'}
            
            return {'valid': True, 'message': 'Configuration is valid'}
            
        except Exception as e:
            return {'valid': False, 'message': str(e)}
    
    async def _test_notification_channel(self, channel: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test a notification channel
        """
        try:
            # Create test alert
            test_alert = {
                'alert_id': 'test-alert',
                'alert_type': 'system_test',
                'model_id': 'test-model',
                'severity': AlertSeverity.INFO.value,
                'title': 'Test Alert - Channel Configuration',
                'description': 'This is a test alert to verify channel configuration.',
                'metadata': {
                    'dashboard_url': '#',
                    'alert_url': '#',
                    'test': True
                },
                'created_at': datetime.utcnow(),
                'tags': ['test']
            }
            
            # Send test notification
            result = await self._send_to_channel(test_alert, channel)
            
            return {
                'success': result['success'],
                'message': 'Test successful' if result['success'] else f"Test failed: {result.get('error')}",
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Test failed with exception: {str(e)}',
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _check_duplicate_alert(self, new_alert: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check for duplicate alerts
        """
        try:
            # Define similarity criteria
            similarity_window = timedelta(minutes=30)
            current_time = datetime.utcnow()
            
            for alert_id, existing_alert in self.active_alerts.items():
                # Skip if alert is too old
                if (current_time - existing_alert['created_at']) > similarity_window:
                    continue
                
                # Check for similarity
                if (existing_alert['model_id'] == new_alert['model_id'] and
                    existing_alert['alert_type'] == new_alert['alert_type'] and
                    existing_alert['severity'] == new_alert['severity']):
                    
                    return existing_alert
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking duplicate alerts: {e}")
            return None
    
    async def _is_alert_suppressed(self, alert: Dict[str, Any]) -> bool:
        """
        Check if alert should be suppressed
        """
        try:
            # Check maintenance windows
            current_time = datetime.utcnow()
            for window in self.maintenance_windows:
                if (window['start_time'] <= current_time <= window['end_time'] and
                    (not window.get('models') or alert['model_id'] in window['models'])):
                    return True
            
            # Check suppression rules
            for rule in self.suppression_rules:
                if self._matches_suppression_rule(alert, rule):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking alert suppression: {e}")
            return False
    
    def _matches_suppression_rule(self, alert: Dict[str, Any], rule: Dict[str, Any]) -> bool:
        """
        Check if alert matches suppression rule
        """
        try:
            # Check model ID
            if 'model_ids' in rule and alert['model_id'] not in rule['model_ids']:
                return False
            
            # Check alert type
            if 'alert_types' in rule and alert['alert_type'] not in rule['alert_types']:
                return False
            
            # Check severity
            if 'severities' in rule and alert['severity'] not in rule['severities']:
                return False
            
            # Check tags
            if 'tags' in rule:
                if not any(tag in alert['tags'] for tag in rule['tags']):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error matching suppression rule: {e}")
            return False
    
    async def _get_target_channels(self, alert: Dict[str, Any]) -> List[str]:
        """
        Get target notification channels for alert
        """
        try:
            # Default: send to all enabled channels
            target_channels = [
                channel_id for channel_id, channel in self.notification_channels.items()
                if channel['enabled']
            ]
            
            # Apply routing rules based on severity, model, etc.
            # This would be configurable in a production system
            if alert['severity'] == AlertSeverity.CRITICAL.value:
                # Critical alerts go to all channels
                pass
            elif alert['severity'] == AlertSeverity.ERROR.value:
                # Error alerts go to primary channels only
                target_channels = [
                    ch_id for ch_id in target_channels
                    if self.notification_channels[ch_id].get('priority', 'normal') in ['high', 'critical']
                ]
            
            return target_channels
            
        except Exception as e:
            logger.error(f"Error getting target channels: {e}")
            return list(self.notification_channels.keys())
    
    async def _send_acknowledgment_notification(self, alert: Dict[str, Any], acknowledged_by: str, notes: Optional[str]):
        """
        Send acknowledgment notification
        """
        try:
            # Create acknowledgment message
            ack_alert = {
                **alert,
                'alert_type': 'acknowledgment',
                'title': f"Alert Acknowledged: {alert['title']}",
                'description': f"Alert has been acknowledged by {acknowledged_by}. {notes or ''}",
                'metadata': {
                    **alert.get('metadata', {}),
                    'acknowledged_by': acknowledged_by,
                    'acknowledgment_notes': notes
                }
            }
            
            # Send to selected channels (typically fewer than original alert)
            target_channels = await self._get_acknowledgment_channels(alert)
            
            for channel_id in target_channels:
                if channel_id in self.notification_channels:
                    channel = self.notification_channels[channel_id]
                    await self._send_to_channel(ack_alert, channel)
            
        except Exception as e:
            logger.error(f"Error sending acknowledgment notification: {e}")
    
    async def _send_resolution_notification(self, alert: Dict[str, Any], resolved_by: str, resolution_notes: Optional[str]):
        """
        Send resolution notification
        """
        try:
            # Create resolution message
            res_alert = {
                **alert,
                'alert_type': 'resolution',
                'title': f"Alert Resolved: {alert['title']}",
                'description': f"Alert has been resolved by {resolved_by}. {resolution_notes or ''}",
                'metadata': {
                    **alert.get('metadata', {}),
                    'resolved_by': resolved_by,
                    'resolution_notes': resolution_notes,
                    'resolution_time_minutes': alert.get('resolution_time_seconds', 0) / 60
                }
            }
            
            # Send to selected channels
            target_channels = await self._get_resolution_channels(alert)
            
            for channel_id in target_channels:
                if channel_id in self.notification_channels:
                    channel = self.notification_channels[channel_id]
                    await self._send_to_channel(res_alert, channel)
            
        except Exception as e:
            logger.error(f"Error sending resolution notification: {e}")
    
    async def _get_acknowledgment_channels(self, alert: Dict[str, Any]) -> List[str]:
        """
        Get channels for acknowledgment notifications
        """
        # Typically send to fewer channels than original alert
        return [
            ch_id for ch_id, channel in self.notification_channels.items()
            if channel['enabled'] and channel.get('acknowledgments', True)
        ]
    
    async def _get_resolution_channels(self, alert: Dict[str, Any]) -> List[str]:
        """
        Get channels for resolution notifications
        """
        # Send to channels that want resolution notifications
        return [
            ch_id for ch_id, channel in self.notification_channels.items()
            if channel['enabled'] and channel.get('resolutions', True)
        ]
    
    async def get_active_alerts(self, model_id: Optional[str] = None, severity: Optional[str] = None) -> Dict[str, Any]:
        """
        Get currently active alerts
        
        Args:
            model_id: Optional filter by model ID
            severity: Optional filter by severity
        
        Returns:
            List of active alerts
        """
        try:
            alerts = []
            
            for alert in self.active_alerts.values():
                # Apply filters
                if model_id and alert['model_id'] != model_id:
                    continue
                if severity and alert['severity'] != severity:
                    continue
                
                # Create summary
                alert_summary = {
                    'alert_id': alert['alert_id'],
                    'alert_type': alert['alert_type'],
                    'model_id': alert['model_id'],
                    'severity': alert['severity'],
                    'status': alert['status'],
                    'title': alert['title'],
                    'created_at': alert['created_at'].isoformat(),
                    'updated_at': alert['updated_at'].isoformat(),
                    'notification_count': alert['notification_count'],
                    'tags': alert['tags']
                }
                alerts.append(alert_summary)
            
            # Sort by creation time (newest first)
            alerts.sort(key=lambda x: x['created_at'], reverse=True)
            
            return {
                'status': 'success',
                'alerts': alerts,
                'total_count': len(alerts),
                'filters_applied': {'model_id': model_id, 'severity': severity}
            }
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def get_alert_statistics(self, time_window: timedelta = timedelta(days=7)) -> Dict[str, Any]:
        """
        Get alert statistics for a time window
        
        Args:
            time_window: Time window for statistics
        
        Returns:
            Alert statistics
        """
        try:
            current_time = datetime.utcnow()
            start_time = current_time - time_window
            
            # Collect statistics from all model histories
            total_alerts = 0
            alerts_by_severity = defaultdict(int)
            alerts_by_type = defaultdict(int)
            alerts_by_model = defaultdict(int)
            resolution_times = []
            
            for model_id, alerts in self.alert_history.items():
                for alert in alerts:
                    if alert['created_at'] >= start_time:
                        total_alerts += 1
                        alerts_by_severity[alert['severity']] += 1
                        alerts_by_type[alert['alert_type']] += 1
                        alerts_by_model[alert['model_id']] += 1
                        
                        if alert.get('resolution_time_seconds'):
                            resolution_times.append(alert['resolution_time_seconds'] / 60)  # Convert to minutes
            
            # Calculate statistics
            stats = {
                'time_window_hours': time_window.total_seconds() / 3600,
                'total_alerts': total_alerts,
                'active_alerts': len(self.active_alerts),
                'alerts_by_severity': dict(alerts_by_severity),
                'alerts_by_type': dict(alerts_by_type),
                'top_models_by_alerts': dict(sorted(alerts_by_model.items(), key=lambda x: x[1], reverse=True)[:10]),
                'resolution_times': {
                    'average_minutes': np.mean(resolution_times) if resolution_times else 0,
                    'median_minutes': np.median(resolution_times) if resolution_times else 0,
                    'min_minutes': min(resolution_times) if resolution_times else 0,
                    'max_minutes': max(resolution_times) if resolution_times else 0
                },
                'notification_channels': {
                    'total_channels': len(self.notification_channels),
                    'enabled_channels': sum(1 for ch in self.notification_channels.values() if ch['enabled']),
                    'channel_success_rates': {
                        ch_id: {
                            'success_rate': ch['success_count'] / max(1, ch['success_count'] + ch['failure_count']),
                            'total_notifications': ch['success_count'] + ch['failure_count']
                        }
                        for ch_id, ch in self.notification_channels.items()
                    }
                }
            }
            
            return {
                'status': 'success',
                'statistics': stats,
                'generated_at': current_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting alert statistics: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

# Global service instance
alerting_service = AlertingService()
