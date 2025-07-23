import React from 'react';
import { motion } from 'framer-motion';
import { CircularProgress, Box, Typography, Skeleton } from '@mui/material';

interface EnterpriseLoadingProps {
  type?: 'skeleton' | 'spinner' | 'pulse' | 'shimmer';
  message?: string;
  progress?: number;
  rows?: number;
  height?: string | number;
  className?: string;
}

const EnterpriseLoadingState: React.FC<EnterpriseLoadingProps> = ({
  type = 'skeleton',
  message = 'Loading...',
  progress,
  rows = 3,
  height = 40,
  className = ''
}) => {
  if (type === 'skeleton') {
    return (
      <div className={`space-y-4 p-6 ${className}`}>
        {Array.from({ length: rows }).map((_, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <Skeleton 
              variant="rectangular" 
              height={height} 
              className="rounded-lg bg-gradient-to-r from-gray-200 to-gray-300"
              animation="wave"
            />
            {index === 0 && (
              <div className="flex space-x-4 mt-3">
                <Skeleton variant="circular" width={48} height={48} />
                <div className="flex-1 space-y-2">
                  <Skeleton variant="text" width="60%" />
                  <Skeleton variant="text" width="40%" />
                </div>
              </div>
            )}
          </motion.div>
        ))}
      </div>
    );
  }

  if (type === 'spinner') {
    return (
      <motion.div 
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className={`flex flex-col items-center justify-center p-12 ${className}`}
      >
        <div className="relative">
          <CircularProgress 
            size={64} 
            thickness={2}
            sx={{ 
              color: 'primary.main',
              '& .MuiCircularProgress-circle': {
                strokeLinecap: 'round',
              }
            }}
            variant={progress ? 'determinate' : 'indeterminate'}
            value={progress}
          />
          {progress && (
            <Box
              sx={{
                top: 0,
                left: 0,
                bottom: 0,
                right: 0,
                position: 'absolute',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Typography variant="caption" component="div" color="text.secondary">
                {`${Math.round(progress)}%`}
              </Typography>
            </Box>
          )}
        </div>
        <Typography 
          variant="h6" 
          className="mt-4 text-gray-700 dark:text-gray-300 font-medium"
        >
          {message}
        </Typography>
      </motion.div>
    );
  }

  if (type === 'pulse') {
    return (
      <motion.div 
        className={`space-y-4 p-6 ${className}`}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        {Array.from({ length: rows }).map((_, index) => (
          <motion.div
            key={index}
            className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-gray-800 dark:to-gray-700 rounded-lg p-4"
            animate={{
              opacity: [0.5, 1, 0.5],
              scale: [0.98, 1, 0.98]
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              delay: index * 0.2
            }}
            style={{ height }}
          />
        ))}
      </motion.div>
    );
  }

  // Shimmer effect
  return (
    <div className={`space-y-4 p-6 ${className}`}>
      {Array.from({ length: rows }).map((_, index) => (
        <div
          key={index}
          className="relative overflow-hidden bg-gray-200 dark:bg-gray-700 rounded-lg animate-pulse"
          style={{ height }}
        >
          <div className="absolute inset-0 -translate-x-full animate-[shimmer_2s_infinite] bg-gradient-to-r from-transparent via-white/60 to-transparent dark:via-gray-500/30" />
        </div>
      ))}
    </div>
  );
};

export default EnterpriseLoadingState;