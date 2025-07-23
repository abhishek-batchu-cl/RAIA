import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Grid,
  MenuItem,
  TextField,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Divider,
} from '@mui/material';
import { Compare, Download } from '@mui/icons-material';
import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from '../../services/api';
import type { Configuration, ComparisonResult } from '../../types/evaluation';

const Comparison: React.FC = () => {
  const [selectedConfigs, setSelectedConfigs] = useState<string[]>([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [comparisonResult, setComparisonResult] = useState<ComparisonResult | null>(null);

  const { data: configurations = [] } = useQuery<Configuration[]>({
    queryKey: ['configurations'],
    queryFn: () => api.getConfigurations(),
  });

  const { data: datasets = [] } = useQuery<string[]>({
    queryKey: ['datasets'],
    queryFn: () => api.getDatasets(),
  });

  const compareMutation = useMutation({
    mutationFn: ({ configIds, dataset }: { configIds: string[]; dataset: string }) =>
      api.compareConfigurations(configIds, dataset),
    onSuccess: (data) => {
      setComparisonResult(data);
    },
  });

  const exportMutation = useMutation({
    mutationFn: (evaluationIds: string[]) => api.exportResults(evaluationIds),
    onSuccess: (blob) => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'comparison_results.csv';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    },
  });

  const handleCompare = () => {
    if (selectedConfigs.length >= 2 && selectedDataset) {
      compareMutation.mutate({
        configIds: selectedConfigs,
        dataset: selectedDataset,
      });
    }
  };

  const handleExport = () => {
    if (comparisonResult) {
      const evaluationIds = comparisonResult.results.map(r => r.id);
      exportMutation.mutate(evaluationIds);
    }
  };

  const getWinnerColor = (configId: string) => {
    if (!comparisonResult?.comparison_metrics.winner) return 'default';
    return comparisonResult.comparison_metrics.winner === configId ? 'success' : 'default';
  };

  const formatPercentage = (value: number) => `${(value * 100).toFixed(1)}%`;
  
  const formatDifference = (value: number) => {
    const formatted = (value * 100).toFixed(1);
    return value > 0 ? `+${formatted}%` : `${formatted}%`;
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Model Comparison
      </Typography>

      <Grid container spacing={3}>
        {/* Comparison Setup */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Setup Comparison
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    select
                    label="Configurations to Compare"
                    value=""
                    helperText={`Selected: ${selectedConfigs.length} configurations`}
                    SelectProps={{
                      multiple: true,
                      value: selectedConfigs,
                      onChange: (e) => setSelectedConfigs(e.target.value as string[]),
                      renderValue: (selected) => 
                        (selected as string[]).map(id => 
                          configurations.find(c => c.id === id)?.name
                        ).join(', ')
                    }}
                  >
                    {configurations.map((config) => (
                      <MenuItem key={config.id} value={config.id}>
                        {config.name} ({config.model_name})
                      </MenuItem>
                    ))}
                  </TextField>
                </Grid>
                
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    select
                    label="Dataset"
                    value={selectedDataset}
                    onChange={(e) => setSelectedDataset(e.target.value)}
                  >
                    {datasets.map((dataset) => (
                      <MenuItem key={dataset} value={dataset}>
                        {dataset}
                      </MenuItem>
                    ))}
                  </TextField>
                </Grid>
                
                <Grid item xs={12}>
                  <Button
                    fullWidth
                    variant="contained"
                    startIcon={<Compare />}
                    onClick={handleCompare}
                    disabled={
                      selectedConfigs.length < 2 ||
                      !selectedDataset ||
                      compareMutation.isPending
                    }
                  >
                    {compareMutation.isPending ? 'Comparing...' : 'Compare Models'}
                  </Button>
                </Grid>
              </Grid>

              {compareMutation.error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {compareMutation.error.message}
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Quick Stats */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Comparison Summary
              </Typography>
              
              {comparisonResult ? (
                <Box>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Winner
                    </Typography>
                    <Typography variant="h6" color="success.main">
                      {configurations.find(c => c.id === comparisonResult.comparison_metrics.winner)?.name || 'Unknown'}
                    </Typography>
                  </Box>
                  
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Configurations Compared
                    </Typography>
                    <Typography variant="body1">
                      {comparisonResult.configurations.length}
                    </Typography>
                  </Box>

                  <Button
                    variant="outlined"
                    startIcon={<Download />}
                    onClick={handleExport}
                    disabled={exportMutation.isPending}
                  >
                    Export Results
                  </Button>
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Run a comparison to see summary
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Comparison Results */}
        {comparisonResult && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Comparison Results
                </Typography>
                
                <TableContainer component={Paper} variant="outlined">
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Configuration</TableCell>
                        <TableCell>Overall Score</TableCell>
                        <TableCell>Accuracy</TableCell>
                        <TableCell>Relevance</TableCell>
                        <TableCell>Coherence</TableCell>
                        <TableCell>Groundedness</TableCell>
                        <TableCell>Fluency</TableCell>
                        <TableCell>Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {comparisonResult.results.map((result) => {
                        const config = comparisonResult.configurations.find(c => c.id === result.configuration_id);
                        return (
                          <TableRow key={result.id}>
                            <TableCell>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Typography variant="body2" fontWeight="medium">
                                  {config?.name}
                                </Typography>
                                <Chip
                                  label={result.configuration_id === comparisonResult.comparison_metrics.winner ? 'Winner' : ''}
                                  color={getWinnerColor(result.configuration_id) as any}
                                  size="small"
                                  sx={{ visibility: result.configuration_id === comparisonResult.comparison_metrics.winner ? 'visible' : 'hidden' }}
                                />
                              </Box>
                              <Typography variant="caption" color="text.secondary">
                                {config?.model_name}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2" fontWeight="medium">
                                {result.results ? formatPercentage(result.results.overall_score) : '-'}
                              </Typography>
                            </TableCell>
                            <TableCell>{result.results ? formatPercentage(result.results.accuracy) : '-'}</TableCell>
                            <TableCell>{result.results ? formatPercentage(result.results.relevance) : '-'}</TableCell>
                            <TableCell>{result.results ? formatPercentage(result.results.coherence) : '-'}</TableCell>
                            <TableCell>{result.results ? formatPercentage(result.results.groundedness) : '-'}</TableCell>
                            <TableCell>{result.results ? formatPercentage(result.results.fluency) : '-'}</TableCell>
                            <TableCell>
                              <Chip
                                label={result.status}
                                color={result.status === 'completed' ? 'success' : result.status === 'failed' ? 'error' : 'warning'}
                                size="small"
                              />
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </TableContainer>

                <Divider sx={{ my: 3 }} />

                <Typography variant="h6" gutterBottom>
                  Performance Differences
                </Typography>
                
                <Grid container spacing={2}>
                  {Object.entries(comparisonResult.comparison_metrics.performance_diff).map(([metric, diff]) => (
                    <Grid item xs={12} sm={6} md={4} key={metric}>
                      <Card variant="outlined">
                        <CardContent sx={{ py: 2 }}>
                          <Typography variant="body2" color="text.secondary" gutterBottom>
                            {metric.charAt(0).toUpperCase() + metric.slice(1)} Difference
                          </Typography>
                          <Typography
                            variant="h6"
                            color={diff > 0 ? 'success.main' : diff < 0 ? 'error.main' : 'text.primary'}
                          >
                            {formatDifference(diff)}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            vs. baseline
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>

                <Box sx={{ mt: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Statistical Significance
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    {Object.entries(comparisonResult.comparison_metrics.statistical_significance).map(([metric, significant]) => (
                      <Chip
                        key={metric}
                        label={`${metric}: ${significant ? 'Significant' : 'Not Significant'}`}
                        color={significant ? 'success' : 'default'}
                        variant="outlined"
                        size="small"
                      />
                    ))}
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default Comparison;
