# RAIA Platform - Responsible AI Analytics

A unified enterprise-scale platform combining comprehensive AI agent evaluation and model explainability capabilities.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Kubernetes (optional, for production)

### Local Development
```bash
# Clone and setup
git clone <repository-url>
cd raia-platform

# Backend setup
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend setup (new terminal)
cd frontend
npm install
npm run dev
```

### Docker Development
```bash
docker-compose up -d
```

## 📋 Features

### Agent Evaluation
- Multi-model agent testing (OpenAI, Anthropic, local models)
- RAG pipeline management with vector search
- Real-time chat interface for interactive testing
- Batch evaluation with comprehensive metrics
- Performance analytics and model comparison

### Model Explainability
- SHAP, LIME, and feature importance analysis
- Individual prediction explanations
- What-if analysis and counterfactuals
- Data drift detection and monitoring
- Fairness and bias analysis

### Enterprise Features
- Multi-tenant architecture with RBAC
- Real-time monitoring and alerting
- Custom dashboard builder
- Compliance reporting (GDPR, CCPA)
- Global deployment with high availability

## 🏗️ Architecture

```
RAIA Platform
├── Backend Services (FastAPI)
│   ├── Authentication Service
│   ├── Agent Evaluation Service
│   ├── Model Explainability Service
│   ├── Data Management Service
│   └── Monitoring Service
├── Frontend (Next.js + React)
│   ├── Executive Dashboard
│   ├── Agent Evaluation UI
│   ├── Model Explainability UI
│   └── Administration Panel
├── Data Layer
│   ├── PostgreSQL (Primary)
│   ├── Redis (Caching)
│   ├── ChromaDB (Vector Search)
│   └── InfluxDB (Time Series)
└── Infrastructure
    ├── Kubernetes Orchestration
    ├── Prometheus Monitoring
    └── ELK Stack Logging
```

## 📚 Documentation

- [Architecture Overview](./docs/ARCHITECTURE.md)
- [API Documentation](./docs/API.md)
- [Deployment Guide](./docs/DEPLOYMENT.md)
- [Development Guide](./docs/DEVELOPMENT.md)
- [User Manual](./docs/USER_GUIDE.md)

## 🔧 Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/raia
REDIS_URL=redis://localhost:6379

# Authentication
JWT_SECRET_KEY=your-secret-key
AUTH_PROVIDER=auth0  # or keycloak

# LLM APIs
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Monitoring
PROMETHEUS_URL=http://localhost:9090
```

## 🧪 Testing

```bash
# Backend tests
cd backend && python -m pytest

# Frontend tests
cd frontend && npm test

# E2E tests
npm run test:e2e
```

## 🚀 Deployment

### Development
```bash
docker-compose -f docker-compose.dev.yml up
```

### Production
```bash
# Using Kubernetes
kubectl apply -f infrastructure/kubernetes/

# Using Docker Compose
docker-compose -f docker-compose.prod.yml up
```

## 📊 Performance Targets

- **Concurrent Users**: 10,000+
- **API Response Time**: < 500ms (95th percentile)
- **Dashboard Load**: < 2 seconds
- **Uptime**: 99.9% availability

## 🤝 Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](./LICENSE) file.

## 🆘 Support

- Documentation: [docs.raia-platform.com](https://docs.raia-platform.com)
- Issues: [GitHub Issues](https://github.com/your-org/raia-platform/issues)
- Support: support@raia-platform.com