import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Timeline,
  Assessment,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { api } from '../../services/api';
import type { MonitoringData } from '../../types/evaluation';

const Monitoring: React.FC = () => {
  const { data: monitoring, isLoading } = useQuery<MonitoringData>({
    queryKey: ['monitoring'],
    queryFn: () => api.getMonitoring(),
    refetchInterval: 10000, // Refresh every 10 seconds
  });

  if (isLoading) {
    return (
      <Box sx={{ width: '100%', mt: 2 }}>
        <LinearProgress />
      </Box>
    );
  }

  const getPerformanceTrend = (trends: number[]) => {
    if (trends.length < 2) return 'stable';
    const latest = trends[trends.length - 1];
    const previous = trends[trends.length - 2];
    return latest > previous ? 'up' : latest < previous ? 'down' : 'stable';
  };

  const accuracyTrend = getPerformanceTrend(monitoring?.performance_trends?.accuracy_trend || []);
  const responseTrend = getPerformanceTrend(monitoring?.performance_trends?.response_time_trend || []);

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Monitoring & Analytics
      </Typography>

      <Grid container spacing={3}>
        {/* Key Metrics */}
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h4" color="primary">
                    {monitoring?.total_evaluations || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Evaluations
                  </Typography>
                </Box>
                <Assessment sx={{ fontSize: 40, color: 'primary.main' }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h4" color="warning.main">
                    {monitoring?.avg_response_time || 0}ms
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Avg Response Time
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    {responseTrend === 'up' ? (
                      <TrendingUp sx={{ fontSize: 16, color: 'error.main' }} />
                    ) : responseTrend === 'down' ? (
                      <TrendingDown sx={{ fontSize: 16, color: 'success.main' }} />
                    ) : (
                      <Timeline sx={{ fontSize: 16, color: 'grey.500' }} />
                    )}
                    <Typography variant="caption" sx={{ ml: 0.5 }}>
                      {responseTrend}
                    </Typography>
                  </Box>
                </Box>
                <Timeline sx={{ fontSize: 40, color: 'warning.main' }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h4" color="success.main">
                    {Math.round((monitoring?.success_rate || 0) * 100)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Success Rate
                  </Typography>
                </Box>
                <TrendingUp sx={{ fontSize: 40, color: 'success.main' }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h4" color="secondary.main">
                    {((monitoring?.performance_trends?.accuracy_trend?.slice(-1)[0] || 0) * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Latest Accuracy
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    {accuracyTrend === 'up' ? (
                      <TrendingUp sx={{ fontSize: 16, color: 'success.main' }} />
                    ) : accuracyTrend === 'down' ? (
                      <TrendingDown sx={{ fontSize: 16, color: 'error.main' }} />
                    ) : (
                      <Timeline sx={{ fontSize: 16, color: 'grey.500' }} />
                    )}
                    <Typography variant="caption" sx={{ ml: 0.5 }}>
                      {accuracyTrend}
                    </Typography>
                  </Box>
                </Box>
                <Assessment sx={{ fontSize: 40, color: 'secondary.main' }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Performance Trends */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Performance Trends
              </Typography>
              
              {monitoring?.performance_trends ? (
                <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Typography variant="body2" color="text.secondary">
                    Chart visualization would be implemented here
                    (using a library like Chart.js or Recharts)
                  </Typography>
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No performance data available
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Activity
              </Typography>
              
              <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
                {monitoring?.recent_activity?.map((activity, index) => (
                  <Box
                    key={index}
                    sx={{
                      pb: 2,
                      mb: 2,
                      borderBottom: index < monitoring.recent_activity.length - 1 ? 1 : 0,
                      borderColor: 'divider',
                    }}
                  >
                    <Typography variant="body2" fontWeight="medium">
                      {activity.action}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {activity.details}
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                      {new Date(activity.timestamp).toLocaleString()}
                    </Typography>
                  </Box>
                )) || (
                  <Typography variant="body2" color="text.secondary">
                    No recent activity
                  </Typography>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* System Health */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Health
              </Typography>
              
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Metric</TableCell>
                      <TableCell>Value</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell>Last Updated</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow>
                      <TableCell>API Response Time</TableCell>
                      <TableCell>{monitoring?.avg_response_time || 0}ms</TableCell>
                      <TableCell>
                        <Chip
                          label={monitoring?.avg_response_time && monitoring.avg_response_time < 1000 ? 'Good' : 'Warning'}
                          color={monitoring?.avg_response_time && monitoring.avg_response_time < 1000 ? 'success' : 'warning'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>{new Date().toLocaleString()}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Success Rate</TableCell>
                      <TableCell>{Math.round((monitoring?.success_rate || 0) * 100)}%</TableCell>
                      <TableCell>
                        <Chip
                          label={monitoring?.success_rate && monitoring.success_rate > 0.9 ? 'Excellent' : 'Good'}
                          color={monitoring?.success_rate && monitoring.success_rate > 0.9 ? 'success' : 'primary'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>{new Date().toLocaleString()}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Total Evaluations</TableCell>
                      <TableCell>{monitoring?.total_evaluations || 0}</TableCell>
                      <TableCell>
                        <Chip label="Active" color="primary" size="small" />
                      </TableCell>
                      <TableCell>{new Date().toLocaleString()}</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Monitoring;
