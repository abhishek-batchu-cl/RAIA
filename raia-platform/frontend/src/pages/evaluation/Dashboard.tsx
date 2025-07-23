import React from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  Assessment,
  Speed,
  CheckCircle,
  TrendingUp,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { api } from '../../services/api';
import type { MonitoringData } from '../../types/evaluation';

const Dashboard: React.FC = () => {
  const { data: monitoring, isLoading } = useQuery<MonitoringData>({
    queryKey: ['monitoring'],
    queryFn: () => api.getMonitoring(),
  });

  if (isLoading) {
    return (
      <Box sx={{ width: '100%', mt: 2 }}>
        <LinearProgress />
      </Box>
    );
  }

  const statCards = [
    {
      title: 'Total Evaluations',
      value: monitoring?.total_evaluations || 0,
      icon: <Assessment sx={{ fontSize: 40, color: 'primary.main' }} />,
      color: 'primary.light',
    },
    {
      title: 'Avg Response Time',
      value: `${monitoring?.avg_response_time || 0}ms`,
      icon: <Speed sx={{ fontSize: 40, color: 'warning.main' }} />,
      color: 'warning.light',
    },
    {
      title: 'Success Rate',
      value: `${Math.round((monitoring?.success_rate || 0) * 100)}%`,
      icon: <CheckCircle sx={{ fontSize: 40, color: 'success.main' }} />,
      color: 'success.light',
    },
    {
      title: 'Performance Trend',
      value: monitoring?.performance_trends?.accuracy_trend.slice(-1)[0]?.toFixed(2) || 'N/A',
      icon: <TrendingUp sx={{ fontSize: 40, color: 'secondary.main' }} />,
      color: 'secondary.light',
    },
  ];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {statCards.map((card, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card
              sx={{
                height: '100%',
                background: `linear-gradient(135deg, ${card.color}20, transparent)`,
                border: 1,
                borderColor: card.color,
              }}
            >
              <CardContent
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                }}
              >
                <Box>
                  <Typography variant="h4" fontWeight="bold">
                    {card.value}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {card.title}
                  </Typography>
                </Box>
                {card.icon}
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Activity
              </Typography>
              {monitoring?.recent_activity?.map((activity, index) => (
                <Box
                  key={index}
                  sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    py: 1,
                    borderBottom: index < monitoring.recent_activity.length - 1 ? 1 : 0,
                    borderColor: 'divider',
                  }}
                >
                  <Box>
                    <Typography variant="body2" fontWeight="medium">
                      {activity.action}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {activity.details}
                    </Typography>
                  </Box>
                  <Typography variant="caption" color="text.secondary">
                    {new Date(activity.timestamp).toLocaleString()}
                  </Typography>
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Actions
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Chip
                  label="Start New Evaluation"
                  color="primary"
                  variant="outlined"
                  clickable
                  sx={{ justifyContent: 'flex-start' }}
                />
                <Chip
                  label="Configure Agent"
                  color="secondary"
                  variant="outlined"
                  clickable
                  sx={{ justifyContent: 'flex-start' }}
                />
                <Chip
                  label="Upload Documents"
                  color="success"
                  variant="outlined"
                  clickable
                  sx={{ justifyContent: 'flex-start' }}
                />
                <Chip
                  label="View Monitoring"
                  color="warning"
                  variant="outlined"
                  clickable
                  sx={{ justifyContent: 'flex-start' }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
