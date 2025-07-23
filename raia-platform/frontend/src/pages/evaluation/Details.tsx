import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import { ExpandMore } from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { useParams } from 'react-router-dom';
import { api } from '../../services/api';
import type { EvaluationResult, Configuration } from '../../types/evaluation';

const Details: React.FC = () => {
  const { evaluationId } = useParams<{ evaluationId: string }>();

  const { data: evaluation, isLoading } = useQuery<EvaluationResult>({
    queryKey: ['evaluation', evaluationId],
    queryFn: () => api.getEvaluation(evaluationId!),
    enabled: !!evaluationId,
  });

  const { data: configuration } = useQuery<Configuration>({
    queryKey: ['configuration', evaluation?.configuration_id],
    queryFn: () => api.getConfiguration(evaluation!.configuration_id),
    enabled: !!evaluation?.configuration_id,
  });

  if (isLoading) {
    return (
      <Box sx={{ width: '100%', mt: 2 }}>
        <LinearProgress />
      </Box>
    );
  }

  if (!evaluation) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Typography variant="h6" color="text.secondary">
          Evaluation not found
        </Typography>
      </Box>
    );
  }

  const formatScore = (score: number) => `${(score * 100).toFixed(1)}%`;

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Evaluation Details
      </Typography>

      <Grid container spacing={3}>
        {/* Overview */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Overview
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    Configuration
                  </Typography>
                  <Typography variant="body1" fontWeight="medium">
                    {configuration?.name || 'Unknown'}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {configuration?.model_name}
                  </Typography>
                </Grid>
                
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    Dataset
                  </Typography>
                  <Typography variant="body1" fontWeight="medium">
                    {evaluation.dataset_name}
                  </Typography>
                </Grid>
                
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    Status
                  </Typography>
                  <Chip
                    label={evaluation.status}
                    color={
                      evaluation.status === 'completed'
                        ? 'success'
                        : evaluation.status === 'failed'
                        ? 'error'
                        : 'warning'
                    }
                    size="small"
                  />
                </Grid>
                
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    Started
                  </Typography>
                  <Typography variant="body1">
                    {new Date(evaluation.created_at).toLocaleString()}
                  </Typography>
                  {evaluation.completed_at && (
                    <Typography variant="caption" color="text.secondary">
                      Completed: {new Date(evaluation.completed_at).toLocaleString()}
                    </Typography>
                  )}
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Scores */}
        {evaluation.results && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Performance Scores
                </Typography>
                
                <Grid container spacing={2}>
                  {Object.entries(evaluation.results).map(([metric, score]) => (
                    <Grid item xs={12} sm={6} md={2} key={metric}>
                      <Card variant="outlined">
                        <CardContent sx={{ textAlign: 'center', py: 2 }}>
                          <Typography variant="h4" color="primary">
                            {formatScore(score as number)}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {metric.charAt(0).toUpperCase() + metric.slice(1).replace('_', ' ')}
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Configuration Details */}
        {configuration && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Configuration Details
                </Typography>
                
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6} md={3}>
                    <Typography variant="body2" color="text.secondary">
                      Model
                    </Typography>
                    <Typography variant="body1">
                      {configuration.model_name}
                    </Typography>
                  </Grid>
                  
                  <Grid item xs={12} sm={6} md={3}>
                    <Typography variant="body2" color="text.secondary">
                      Temperature
                    </Typography>
                    <Typography variant="body1">
                      {configuration.temperature}
                    </Typography>
                  </Grid>
                  
                  <Grid item xs={12} sm={6} md={3}>
                    <Typography variant="body2" color="text.secondary">
                      Max Tokens
                    </Typography>
                    <Typography variant="body1">
                      {configuration.max_tokens}
                    </Typography>
                  </Grid>
                  
                  <Grid item xs={12} sm={6} md={3}>
                    <Typography variant="body2" color="text.secondary">
                      Created
                    </Typography>
                    <Typography variant="body1">
                      {new Date(configuration.created_at).toLocaleDateString()}
                    </Typography>
                  </Grid>
                  
                  <Grid item xs={12}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      System Prompt
                    </Typography>
                    <Paper variant="outlined" sx={{ p: 2, backgroundColor: 'grey.50' }}>
                      <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                        {configuration.system_prompt}
                      </Typography>
                    </Paper>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Detailed Results */}
        {evaluation.detailed_results && evaluation.detailed_results.length > 0 && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Detailed Results ({evaluation.detailed_results.length} questions)
                </Typography>
                
                <Box sx={{ maxHeight: 600, overflow: 'auto' }}>
                  {evaluation.detailed_results.map((result, index) => (
                    <Accordion key={index}>
                      <AccordionSummary expandIcon={<ExpandMore />}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
                          <Typography variant="body2" fontWeight="medium">
                            Question {index + 1}
                          </Typography>
                          <Box sx={{ display: 'flex', gap: 1 }}>
                            {Object.entries(result.scores).map(([metric, score]) => (
                              <Chip
                                key={metric}
                                label={`${metric}: ${formatScore(score as number)}`}
                                size="small"
                                variant="outlined"
                                color={score as number > 0.8 ? 'success' : score as number > 0.6 ? 'warning' : 'error'}
                              />
                            ))}
                          </Box>
                        </Box>
                      </AccordionSummary>
                      <AccordionDetails>
                        <Grid container spacing={2}>
                          <Grid item xs={12}>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              Question
                            </Typography>
                            <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
                              <Typography variant="body2">
                                {result.question}
                              </Typography>
                            </Paper>
                          </Grid>
                          
                          <Grid item xs={12} md={6}>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              Expected Answer
                            </Typography>
                            <Paper variant="outlined" sx={{ p: 2, backgroundColor: 'success.50' }}>
                              <Typography variant="body2">
                                {result.expected_answer}
                              </Typography>
                            </Paper>
                          </Grid>
                          
                          <Grid item xs={12} md={6}>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              Actual Answer
                            </Typography>
                            <Paper variant="outlined" sx={{ p: 2, backgroundColor: 'grey.50' }}>
                              <Typography variant="body2">
                                {result.actual_answer}
                              </Typography>
                            </Paper>
                          </Grid>
                          
                          <Grid item xs={12}>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              Scores
                            </Typography>
                            <TableContainer component={Paper} variant="outlined">
                              <Table size="small">
                                <TableHead>
                                  <TableRow>
                                    {Object.keys(result.scores).map((metric) => (
                                      <TableCell key={metric}>
                                        {metric.charAt(0).toUpperCase() + metric.slice(1)}
                                      </TableCell>
                                    ))}
                                  </TableRow>
                                </TableHead>
                                <TableBody>
                                  <TableRow>
                                    {Object.values(result.scores).map((score, i) => (
                                      <TableCell key={i}>
                                        {formatScore(score as number)}
                                      </TableCell>
                                    ))}
                                  </TableRow>
                                </TableBody>
                              </Table>
                            </TableContainer>
                          </Grid>
                        </Grid>
                      </AccordionDetails>
                    </Accordion>
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default Details;
