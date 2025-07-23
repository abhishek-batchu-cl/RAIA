import React from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, Typography, Avatar, Chip, LinearProgress } from '@mui/material';
import { 
  TrendingUp, 
  TrendingDown, 
  Minus,
  AlertTriangle,
  CheckCircle,
  Clock,
  MoreHorizontal
} from 'lucide-react';

interface EnterpriseMetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: {
    direction: 'up' | 'down' | 'neutral';
    value: string | number;
    period?: string;
  };
  status?: 'success' | 'warning' | 'error' | 'info' | 'neutral';
  progress?: {
    value: number;
    max?: number;
    label?: string;
  };
  icon?: React.ReactNode;
  color?: 'blue' | 'green' | 'red' | 'orange' | 'purple' | 'gray';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  actionButton?: {
    label: string;
    onClick: () => void;
  };
  className?: string;
}

const EnterpriseMetricCard: React.FC<EnterpriseMetricCardProps> = ({
  title,
  value,
  subtitle,
  trend,
  status = 'neutral',
  progress,
  icon,
  color = 'blue',
  size = 'md',
  loading = false,
  actionButton,
  className = ''
}) => {
  const colorMap = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    red: 'bg-red-500',
    orange: 'bg-orange-500',
    purple: 'bg-purple-500',
    gray: 'bg-gray-500'
  };

  const statusColors = {
    success: 'text-green-600 bg-green-50 border-green-200',
    warning: 'text-orange-600 bg-orange-50 border-orange-200',
    error: 'text-red-600 bg-red-50 border-red-200',
    info: 'text-blue-600 bg-blue-50 border-blue-200',
    neutral: 'text-gray-600 bg-gray-50 border-gray-200'
  };

  const statusIcons = {
    success: CheckCircle,
    warning: AlertTriangle,
    error: AlertTriangle,
    info: Clock,
    neutral: Minus
  };

  const sizeClasses = {
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8'
  };

  const TrendIcon = trend?.direction === 'up' ? TrendingUp : 
                   trend?.direction === 'down' ? TrendingDown : Minus;
  
  const StatusIcon = statusIcons[status];

  if (loading) {
    return (
      <Card className={`${sizeClasses[size]} ${className} animate-pulse`}>
        <CardContent>
          <div className="space-y-3">
            <div className="h-4 bg-gray-200 rounded w-3/4"></div>
            <div className="h-8 bg-gray-200 rounded w-1/2"></div>
            <div className="h-3 bg-gray-200 rounded w-full"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ 
        y: -2, 
        boxShadow: '0 10px 40px -10px rgba(0,0,0,0.1)' 
      }}
      transition={{ duration: 0.2 }}
    >
      <Card 
        className={`${sizeClasses[size]} ${className} border-l-4 ${
          color === 'blue' ? 'border-l-blue-500' :
          color === 'green' ? 'border-l-green-500' :
          color === 'red' ? 'border-l-red-500' :
          color === 'orange' ? 'border-l-orange-500' :
          color === 'purple' ? 'border-l-purple-500' :
          'border-l-gray-500'
        } hover:shadow-lg transition-shadow duration-200`}
        elevation={1}
      >
        <CardContent className="relative">
          {/* Header */}
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center space-x-3">
              {icon && (
                <Avatar 
                  className={`${colorMap[color]} text-white`}
                  sx={{ width: size === 'lg' ? 48 : size === 'md' ? 40 : 32, height: size === 'lg' ? 48 : size === 'md' ? 40 : 32 }}
                >
                  {icon}
                </Avatar>
              )}
              <div>
                <Typography 
                  variant={size === 'lg' ? 'h6' : 'subtitle2'} 
                  className="font-semibold text-gray-700 dark:text-gray-300"
                >
                  {title}
                </Typography>
                {subtitle && (
                  <Typography 
                    variant="caption" 
                    className="text-gray-500 dark:text-gray-400"
                  >
                    {subtitle}
                  </Typography>
                )}
              </div>
            </div>

            {/* Status Indicator */}
            <Chip
              icon={<StatusIcon size={14} />}
              label={status.charAt(0).toUpperCase() + status.slice(1)}
              size="small"
              className={`${statusColors[status]} border`}
              variant="outlined"
            />
          </div>

          {/* Main Value */}
          <div className="mb-4">
            <Typography 
              variant={size === 'lg' ? 'h3' : size === 'md' ? 'h4' : 'h5'}
              className="font-bold text-gray-900 dark:text-gray-100 mb-1"
            >
              {value}
            </Typography>

            {/* Trend */}
            {trend && (
              <div className="flex items-center space-x-2">
                <div className={`flex items-center space-x-1 px-2 py-1 rounded-full text-xs font-medium ${
                  trend.direction === 'up' ? 'bg-green-100 text-green-700' :
                  trend.direction === 'down' ? 'bg-red-100 text-red-700' :
                  'bg-gray-100 text-gray-700'
                }`}>
                  <TrendIcon size={12} />
                  <span>{trend.value}</span>
                </div>
                {trend.period && (
                  <Typography variant="caption" className="text-gray-500">
                    {trend.period}
                  </Typography>
                )}
              </div>
            )}
          </div>

          {/* Progress Bar */}
          {progress && (
            <div className="mb-4">
              <div className="flex justify-between items-center mb-1">
                <Typography variant="caption" className="text-gray-600 font-medium">
                  {progress.label || 'Progress'}
                </Typography>
                <Typography variant="caption" className="text-gray-500">
                  {progress.value}{progress.max ? `/${progress.max}` : '%'}
                </Typography>
              </div>
              <LinearProgress
                variant="determinate"
                value={(progress.value / (progress.max || 100)) * 100}
                className="rounded-full h-2"
                sx={{
                  backgroundColor: 'rgba(0,0,0,0.1)',
                  '& .MuiLinearProgress-bar': {
                    borderRadius: '4px',
                    backgroundColor: colorMap[color].replace('bg-', '')
                  }
                }}
              />
            </div>
          )}

          {/* Action Button */}
          {actionButton && (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={actionButton.onClick}
              className={`w-full mt-4 px-4 py-2 bg-${color}-500 hover:bg-${color}-600 text-white rounded-lg font-medium text-sm transition-colors duration-200`}
            >
              {actionButton.label}
            </motion.button>
          )}

          {/* More Options */}
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            className="absolute top-4 right-4 p-1 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          >
            <MoreHorizontal size={16} className="text-gray-400" />
          </motion.button>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default EnterpriseMetricCard;