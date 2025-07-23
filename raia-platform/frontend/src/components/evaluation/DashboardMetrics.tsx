import React from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  LinearProgress,
  Chip,
} from '@mui/material';
import {
  Assessment,
  Speed,
  CheckCircle,
  TrendingUp,
  Memory,
  AccessTime,
} from '@mui/icons-material';

interface MetricCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color: string;
  trend?: {
    value: number;
    isPositive: boolean;
  };
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, icon, color, trend }) => (
  <Card sx={{ height: '100%', background: `linear-gradient(135deg, ${color}20 0%, ${color}10 100%)` }}>
    <CardContent>
      <Box display="flex" alignItems="center" justifyContent="space-between">
        <Box>
          <Typography variant="h4" component="div" fontWeight="bold">
            {value}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            {title}
          </Typography>
          {trend && (
            <Box display="flex" alignItems="center" mt={1}>
              <TrendingUp 
                sx={{ 
                  fontSize: 16, 
                  color: trend.isPositive ? 'success.main' : 'error.main',
                  transform: trend.isPositive ? 'none' : 'rotate(180deg)'
                }} 
              />
              <Typography 
                variant="caption" 
                color={trend.isPositive ? 'success.main' : 'error.main'}
                sx={{ ml: 0.5 }}
              >
                {trend.value}%
              </Typography>
            </Box>
          )}
        </Box>
        <Box sx={{ color }}>
          {icon}
        </Box>
      </Box>
    </CardContent>
  </Card>
);

interface DashboardMetricsProps {
  metrics: {
    totalEvaluations: number;
    avgResponseTime: number;
    successRate: number;
    totalTokens: number;
    avgTokensPerQuery: number;
    activeConfigurations: number;
  };
}

const DashboardMetrics: React.FC<DashboardMetricsProps> = ({ metrics }) => {
  const metricCards = [
    {
      title: 'Total Evaluations',
      value: metrics.totalEvaluations.toLocaleString(),
      icon: <Assessment sx={{ fontSize: 40 }} />,
      color: '#1976d2',
      trend: { value: 12.5, isPositive: true }
    },
    {
      title: 'Avg Response Time',
      value: `${metrics.avgResponseTime}ms`,
      icon: <Speed sx={{ fontSize: 40 }} />,
      color: '#ed6c02',
      trend: { value: 5.2, isPositive: false }
    },
    {
      title: 'Success Rate',
      value: `${(metrics.successRate * 100).toFixed(1)}%`,
      icon: <CheckCircle sx={{ fontSize: 40 }} />,
      color: '#2e7d32',
      trend: { value: 2.1, isPositive: true }
    },
    {
      title: 'Total Tokens Used',
      value: metrics.totalTokens.toLocaleString(),
      icon: <Memory sx={{ fontSize: 40 }} />,
      color: '#9c27b0',
      trend: { value: 8.7, isPositive: true }
    },
    {
      title: 'Avg Tokens/Query',
      value: Math.round(metrics.avgTokensPerQuery).toLocaleString(),
      icon: <AccessTime sx={{ fontSize: 40 }} />,
      color: '#d32f2f',
    },
    {
      title: 'Active Configurations',
      value: metrics.activeConfigurations,
      icon: <Assessment sx={{ fontSize: 40 }} />,
      color: '#1976d2',
    },
  ];

  return (
    <Grid container spacing={3}>
      {metricCards.map((card, index) => (
        <Grid item xs={12} sm={6} md={4} key={index}>
          <MetricCard {...card} />
        </Grid>
      ))}
    </Grid>
  );
};

export default DashboardMetrics;
