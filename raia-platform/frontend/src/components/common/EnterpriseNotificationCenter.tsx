import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  IconButton,
  Badge,
  Drawer,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Chip,
  Divider,
  Button,
  Box
} from '@mui/material';
import {
  Bell,
  AlertTriangle,
  CheckCircle,
  Info,
  X,
  Settings,
  Check,
  Trash2
} from 'lucide-react';

interface Notification {
  id: string;
  title: string;
  message: string;
  type: 'success' | 'warning' | 'error' | 'info';
  timestamp: Date;
  read: boolean;
  actionUrl?: string;
  actionLabel?: string;
}

interface EnterpriseNotificationCenterProps {
  notifications?: Notification[];
  onNotificationClick?: (notification: Notification) => void;
  onMarkAsRead?: (notificationId: string) => void;
  onMarkAllAsRead?: () => void;
  onDelete?: (notificationId: string) => void;
  onClearAll?: () => void;
}

const defaultNotifications: Notification[] = [
  {
    id: '1',
    title: 'Model Drift Detected',
    message: 'Credit Risk model showing 15% performance degradation',
    type: 'warning',
    timestamp: new Date(Date.now() - 5 * 60 * 1000),
    read: false,
    actionUrl: '/model-drift',
    actionLabel: 'View Details'
  },
  {
    id: '2',
    title: 'Training Completed',
    message: 'Customer Churn model v2.1 training successful',
    type: 'success',
    timestamp: new Date(Date.now() - 30 * 60 * 1000),
    read: false,
    actionUrl: '/model-overview',
    actionLabel: 'View Model'
  },
  {
    id: '3',
    title: 'Bias Alert',
    message: 'Fairness metrics below threshold for age group 25-35',
    type: 'error',
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
    read: true,
    actionUrl: '/fairness',
    actionLabel: 'Review'
  },
  {
    id: '4',
    title: 'Data Quality Check',
    message: 'Weekly data quality report available',
    type: 'info',
    timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000),
    read: false,
    actionUrl: '/data-quality',
    actionLabel: 'View Report'
  }
];

const EnterpriseNotificationCenter: React.FC<EnterpriseNotificationCenterProps> = ({
  notifications = defaultNotifications,
  onNotificationClick,
  onMarkAsRead,
  onMarkAllAsRead,
  onDelete,
  onClearAll
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const unreadCount = notifications.filter(n => !n.read).length;

  const getNotificationIcon = (type: Notification['type']) => {
    switch (type) {
      case 'success': return <CheckCircle className="text-green-500" />;
      case 'warning': return <AlertTriangle className="text-orange-500" />;
      case 'error': return <AlertTriangle className="text-red-500" />;
      case 'info': return <Info className="text-blue-500" />;
    }
  };

  const getNotificationColor = (type: Notification['type']) => {
    switch (type) {
      case 'success': return 'success';
      case 'warning': return 'warning';
      case 'error': return 'error';
      case 'info': return 'info';
    }
  };

  const formatTimestamp = (timestamp: Date) => {
    const now = new Date();
    const diff = now.getTime() - timestamp.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return `${days}d ago`;
  };

  const handleNotificationClick = (notification: Notification) => {
    if (!notification.read) {
      onMarkAsRead?.(notification.id);
    }
    onNotificationClick?.(notification);
  };

  return (
    <>
      {/* Notification Bell */}
      <motion.div
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <IconButton
          onClick={() => setIsOpen(true)}
          className="relative"
          size="large"
        >
          <Badge 
            badgeContent={unreadCount} 
            color="error"
            variant={unreadCount > 0 ? 'standard' : 'dot'}
          >
            <Bell size={24} className="text-gray-600 dark:text-gray-300" />
          </Badge>
        </IconButton>
      </motion.div>

      {/* Notification Drawer */}
      <Drawer
        anchor="right"
        open={isOpen}
        onClose={() => setIsOpen(false)}
        PaperProps={{
          sx: { width: 420 }
        }}
      >
        <motion.div
          initial={{ x: 100, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          className="h-full flex flex-col"
        >
          {/* Header */}
          <div className="p-4 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
            <div className="flex items-center justify-between mb-3">
              <Typography variant="h6" className="font-semibold">
                Notifications
              </Typography>
              <IconButton 
                onClick={() => setIsOpen(false)}
                size="small"
              >
                <X size={20} />
              </IconButton>
            </div>
            
            <div className="flex items-center justify-between">
              <Typography variant="body2" color="textSecondary">
                {unreadCount} unread notifications
              </Typography>
              <div className="flex space-x-2">
                {unreadCount > 0 && (
                  <Button
                    size="small"
                    startIcon={<Check size={16} />}
                    onClick={onMarkAllAsRead}
                    variant="outlined"
                  >
                    Mark all read
                  </Button>
                )}
                <Button
                  size="small"
                  startIcon={<Settings size={16} />}
                  variant="text"
                >
                  Settings
                </Button>
              </div>
            </div>
          </div>

          {/* Notifications List */}
          <div className="flex-1 overflow-y-auto">
            {notifications.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-64">
                <Bell size={48} className="text-gray-300 mb-4" />
                <Typography variant="body1" color="textSecondary">
                  No notifications yet
                </Typography>
              </div>
            ) : (
              <List className="p-0">
                <AnimatePresence>
                  {notifications.map((notification, index) => (
                    <motion.div
                      key={notification.id}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      transition={{ delay: index * 0.05 }}
                    >
                      <ListItem
                        className={`cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700 ${
                          !notification.read ? 'bg-blue-50/50 dark:bg-blue-900/10' : ''
                        }`}
                        onClick={() => handleNotificationClick(notification)}
                      >
                        <ListItemAvatar>
                          <Avatar
                            className={`${
                              notification.type === 'success' ? 'bg-green-100 text-green-600' :
                              notification.type === 'warning' ? 'bg-orange-100 text-orange-600' :
                              notification.type === 'error' ? 'bg-red-100 text-red-600' :
                              'bg-blue-100 text-blue-600'
                            }`}
                          >
                            {getNotificationIcon(notification.type)}
                          </Avatar>
                        </ListItemAvatar>
                        
                        <ListItemText
                          primary={
                            <div className="flex items-center justify-between">
                              <Typography
                                variant="subtitle2"
                                className={`font-medium ${!notification.read ? 'font-semibold' : ''}`}
                              >
                                {notification.title}
                              </Typography>
                              <div className="flex items-center space-x-2">
                                {!notification.read && (
                                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                                )}
                                <IconButton
                                  size="small"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    onDelete?.(notification.id);
                                  }}
                                  className="opacity-0 hover:opacity-100 transition-opacity"
                                >
                                  <Trash2 size={14} />
                                </IconButton>
                              </div>
                            </div>
                          }
                          secondary={
                            <div className="mt-1">
                              <Typography variant="body2" color="textSecondary" className="mb-2">
                                {notification.message}
                              </Typography>
                              <div className="flex items-center justify-between">
                                <Chip
                                  label={notification.type.charAt(0).toUpperCase() + notification.type.slice(1)}
                                  size="small"
                                  color={getNotificationColor(notification.type)}
                                  variant="outlined"
                                />
                                <Typography variant="caption" color="textSecondary">
                                  {formatTimestamp(notification.timestamp)}
                                </Typography>
                              </div>
                              {notification.actionLabel && (
                                <Button
                                  size="small"
                                  variant="text"
                                  color="primary"
                                  className="mt-2 p-0 min-w-0"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    // Handle action click
                                  }}
                                >
                                  {notification.actionLabel}
                                </Button>
                              )}
                            </div>
                          }
                        />
                      </ListItem>
                      {index < notifications.length - 1 && <Divider />}
                    </motion.div>
                  ))}
                </AnimatePresence>
              </List>
            )}
          </div>

          {/* Footer */}
          {notifications.length > 0 && (
            <div className="p-4 border-t border-gray-200 dark:border-gray-700">
              <Button
                fullWidth
                variant="outlined"
                startIcon={<Trash2 size={16} />}
                onClick={onClearAll}
                className="text-gray-600 border-gray-300 hover:bg-gray-50"
              >
                Clear All Notifications
              </Button>
            </div>
          )}
        </motion.div>
      </Drawer>
    </>
  );
};

export default EnterpriseNotificationCenter;