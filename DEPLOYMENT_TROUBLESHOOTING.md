# AI Trading Agent - Deployment Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide addresses the persistent Kubernetes deployment failures encountered with the AI Trading Agent on Google Cloud Platform. After analyzing the deployment patterns and common failure modes, we have identified several critical issues that prevent pods from becoming ready, even with extended timeout periods.

## Root Cause Analysis

### Primary Issues Identified

The deployment failures stem from several interconnected problems that compound to create a cascade of startup failures. The most critical issues include improper dependency management, inadequate health check configurations, insufficient startup time allowances, and missing initialization sequences.

### Dependency Chain Problems

The original deployment configuration attempted to start all services simultaneously without proper dependency ordering. The backend application requires both PostgreSQL and Redis to be fully operational before it can successfully initialize. However, the deployment manifests did not include proper init containers or startup probes to ensure this dependency chain was respected.

### Health Check Configuration Issues

The original health check configurations were too aggressive, with short timeout periods and insufficient failure thresholds. This caused pods to be marked as failed before they had adequate time to complete their initialization sequences, particularly for the backend application which needs to establish database connections and load AI models.

### Resource Allocation Problems

The resource requests and limits were not optimally configured for the startup phase. While the running application might operate efficiently within the specified limits, the initialization phase requires additional CPU and memory resources to load dependencies, establish connections, and initialize AI models.

## Comprehensive Fixes Implemented

### Enhanced Dockerfile Configuration

The fixed Dockerfile (`Dockerfile_fixed`) includes several critical improvements:

- **Extended Health Check Parameters**: The health check now includes a 60-second start period with 5 retries, allowing adequate time for application initialization.
- **Startup Script Implementation**: A dedicated startup script provides better control over the initialization sequence and includes proper error handling.
- **Dependency Installation**: Additional system dependencies like `postgresql-client` are installed to support database connectivity testing.
- **Directory Structure**: All necessary directories are created during the build process to prevent runtime errors.

### Improved Kubernetes Deployment Configuration

The enhanced backend deployment (`backend-deployment-fixed-v2.yaml`) addresses multiple startup issues:

#### Init Container Implementation

An init container using `busybox` is now included to verify that both PostgreSQL and Redis services are accessible before the main application container starts. This init container uses `nc` (netcat) to test connectivity to the required services, ensuring the dependency chain is properly respected.

#### Enhanced Probe Configuration

The probe configuration has been significantly improved:

- **Startup Probe**: Allows up to 2 minutes (12 failures × 10-second periods) for the application to become ready
- **Readiness Probe**: Configured with appropriate delays and failure thresholds to prevent premature pod restarts
- **Liveness Probe**: Set with longer intervals to avoid false positives during normal operation

#### Cloud SQL Proxy Integration

The Cloud SQL Proxy sidecar container includes its own readiness probe to ensure database connectivity is established before the main application attempts to connect.

### PostgreSQL Deployment Enhancements

The fixed PostgreSQL deployment (`postgres-deployment-fixed.yaml`) includes:

#### Improved Startup Configuration

- **PGDATA Environment Variable**: Properly configured to prevent data directory conflicts
- **Enhanced Initialization Script**: Comprehensive database schema creation with proper permissions
- **Startup Probe Configuration**: Allows up to 100 seconds for PostgreSQL to become ready

#### Resource Optimization

Resource requests and limits have been adjusted to provide adequate resources during the startup phase while maintaining efficiency during normal operation.

### Redis Configuration Improvements

The Redis deployment has been enhanced with:

- **Memory Management**: Configured with `maxmemory` policies to prevent out-of-memory conditions
- **Authentication Configuration**: Proper password-based authentication setup
- **Health Check Optimization**: Improved probe commands that properly authenticate with Redis

## Deployment Script Enhancements

### Version 3 Improvements

The `deploy_fixed_v3.sh` script includes several critical enhancements:

#### Sequential Deployment Strategy

Instead of deploying all services simultaneously, the script now follows a sequential approach:

1. Deploy PostgreSQL and wait for readiness
2. Deploy Redis and wait for readiness  
3. Deploy backend application with dependency verification
4. Deploy frontend application

#### Enhanced Error Handling

The script now includes comprehensive error handling with informative messages and continues execution even if individual wait commands timeout, allowing for manual verification.

#### Extended Timeout Periods

Timeout periods have been significantly extended:
- PostgreSQL: 600 seconds (10 minutes)
- Redis: 600 seconds (10 minutes)
- Backend: 1200 seconds (20 minutes)
- Frontend: 600 seconds (10 minutes)

## Troubleshooting Commands

### Essential Diagnostic Commands

When deployment issues occur, the following commands provide critical diagnostic information:

```bash
# Check pod status and events
kubectl get pods -n ai-trading
kubectl describe pod <pod-name> -n ai-trading

# View application logs
kubectl logs <pod-name> -n ai-trading
kubectl logs <pod-name> -c <container-name> -n ai-trading

# Check service connectivity
kubectl exec -it <pod-name> -n ai-trading -- nc -z postgres-service 5432
kubectl exec -it <pod-name> -n ai-trading -- nc -z redis-service 6379

# Monitor deployment progress
kubectl get events -n ai-trading --sort-by='.lastTimestamp'
```

### Common Failure Patterns

#### Database Connection Failures

If the backend pod fails to start with database connection errors:

1. Verify PostgreSQL pod is running and ready
2. Check database credentials in secrets
3. Verify Cloud SQL Proxy sidecar is functioning
4. Test database connectivity from within the pod

#### Redis Connection Issues

For Redis connectivity problems:

1. Confirm Redis pod status and readiness
2. Verify Redis password configuration
3. Test Redis connectivity using redis-cli
4. Check network policies and service configurations

#### Resource Constraints

If pods are killed due to resource constraints:

1. Review resource requests and limits
2. Check node capacity and available resources
3. Consider scaling cluster nodes if necessary
4. Monitor memory and CPU usage patterns

## Best Practices for Future Deployments

### Dependency Management

Always implement proper dependency ordering using init containers or startup probes. Never assume that services will be available immediately upon deployment.

### Health Check Configuration

Configure health checks with realistic timeouts and failure thresholds. Consider the actual startup time requirements of your applications, including time needed for:

- Dependency loading
- Database connection establishment
- Model initialization
- Cache warming

### Resource Planning

Allocate sufficient resources for both startup and runtime phases. Startup phases often require more resources than steady-state operation.

### Monitoring and Observability

Implement comprehensive logging and monitoring from the beginning. This includes:

- Application startup logs
- Dependency connectivity logs
- Resource utilization metrics
- Health check status tracking

## Recovery Procedures

### Failed Deployment Recovery

If a deployment fails completely:

1. Delete the failed deployment: `kubectl delete deployment <deployment-name> -n ai-trading`
2. Check and fix any configuration issues
3. Redeploy using the fixed configurations
4. Monitor the startup process closely

### Partial Failure Recovery

For partially successful deployments:

1. Identify which components are failing
2. Scale down problematic deployments to zero
3. Fix configuration issues
4. Scale back up gradually
5. Verify each component before proceeding

### Data Recovery

If database initialization fails:

1. Check PostgreSQL logs for specific errors
2. Verify database credentials and permissions
3. Manually connect to the database to test connectivity
4. Re-run initialization scripts if necessary

## Conclusion

The deployment failures were primarily caused by inadequate dependency management, aggressive health check configurations, and insufficient startup time allowances. The comprehensive fixes implemented address these issues through proper init containers, enhanced probe configurations, sequential deployment strategies, and extended timeout periods.

The new deployment configuration should successfully deploy all components without the timeout errors previously encountered. However, it's important to monitor the deployment process and be prepared to adjust timeout values based on actual performance characteristics in your specific environment.

