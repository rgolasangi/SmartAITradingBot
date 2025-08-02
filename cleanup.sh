#!/bin/bash

# AI Trading Agent GCP Cleanup Script
# This script automates the deletion of all Google Cloud resources
# created by the AI Trading Agent deployment script.

set -e  # Exit on any error



# Configuration (replace with your actual project ID)
PROJECT_ID=${1:-"your-gcp-project-id"}
CLUSTER_NAME="ai-trading-cluster"
CLUSTER_ZONE="us-central1-a"
NAMESPACE="ai-trading"
SERVICE_ACCOUNT_NAME="ai-trading-sa"
CLOUD_SQL_INSTANCE="ai-trading-postgres"
REDIS_INSTANCE="ai-trading-redis"

echo -e "${BLUE}=== AI Trading Agent GCP Cleanup Script ===${NC}"
echo -e "${BLUE}Project ID: ${PROJECT_ID}${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if project ID is provided
if [ "$PROJECT_ID" == "your-gcp-project-id" ]; then
    print_error "Please provide your GCP Project ID as the first argument:"
    print_error "Usage: ./cleanup.sh YOUR_GCP_PROJECT_ID"
    exit 1
fi

# Set gcloud project
gcloud config set project $PROJECT_ID

# Delete Kubernetes Namespace
delete_k8s_namespace() {
    print_status "Deleting Kubernetes namespace: $NAMESPACE..."
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        kubectl delete namespace $NAMESPACE --wait=true || print_warning "Failed to delete namespace $NAMESPACE. It might be stuck in terminating state. You may need to delete it manually."
        print_status "Kubernetes namespace $NAMESPACE deleted."
    else
        print_warning "Kubernetes namespace $NAMESPACE does not exist. Skipping deletion."
    fi
}

# Delete GKE Cluster
delete_gke_cluster() {
    print_status "Deleting GKE cluster: $CLUSTER_NAME in $CLUSTER_ZONE..."
    if gcloud container clusters describe $CLUSTER_NAME --zone=$CLUSTER_ZONE &> /dev/null; then
        gcloud container clusters delete $CLUSTER_NAME --zone=$CLUSTER_ZONE --quiet --async
        print_status "GKE cluster deletion initiated. It may take some time to fully delete."
    else
        print_warning "GKE cluster $CLUSTER_NAME does not exist. Skipping deletion."
    fi
}

# Delete Cloud SQL Instance
delete_cloud_sql() {
    print_status "Deleting Cloud SQL instance: $CLOUD_SQL_INSTANCE..."
    if gcloud sql instances describe $CLOUD_SQL_INSTANCE &> /dev/null; then
        gcloud sql instances delete $CLOUD_SQL_INSTANCE --quiet
        print_status "Cloud SQL instance $CLOUD_SQL_INSTANCE deleted."
    else
        print_warning "Cloud SQL instance $CLOUD_SQL_INSTANCE does not exist. Skipping deletion."
    fi
}

# Delete Redis Instance
delete_redis_instance() {
    print_status "Deleting Redis instance: $REDIS_INSTANCE..."
    if gcloud redis instances describe $REDIS_INSTANCE --region=us-central1 &> /dev/null; then
        gcloud redis instances delete $REDIS_INSTANCE --region=us-central1 --quiet
        print_status "Redis instance $REDIS_INSTANCE deleted."
    else
        print_warning "Redis instance $REDIS_INSTANCE does not exist. Skipping deletion."
    fi
}

# Delete Service Account and Key
delete_service_account() {
    print_status "Deleting service account: $SERVICE_ACCOUNT_NAME..."
    if gcloud iam service-accounts describe $SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com &> /dev/null; then
        # Delete service account key first
        if [ -f "./cloud-sql-key.json" ]; then
            print_status "Deleting local service account key file: cloud-sql-key.json"
            rm ./cloud-sql-key.json
        fi
        
        # List and delete all keys associated with the service account
        print_status "Deleting all service account keys for $SERVICE_ACCOUNT_NAME..."
        gcloud iam service-accounts keys list --iam-account=$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com --format="value(name)" | while read -r key_id; do
            gcloud iam service-accounts keys delete "$key_id" --iam-account=$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com --quiet
            print_status "Deleted key: $key_id"
        done

        # Delete the service account
        gcloud iam service-accounts delete $SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com --quiet
        print_status "Service account $SERVICE_ACCOUNT_NAME deleted."
    else
        print_warning "Service account $SERVICE_ACCOUNT_NAME does not exist. Skipping deletion."
    fi
}

# Main cleanup sequence
main() {
    print_status "Starting cleanup process..."
    
    # It's generally safer to delete Kubernetes resources first, then the cluster, and then other GCP resources.
    # This ensures that no pods are left trying to access non-existent resources.
    
    delete_k8s_namespace
    delete_cloud_sql
    delete_redis_instance
    delete_gke_cluster
    delete_service_account
    
    print_status "Cleanup process completed. Some resources (like GKE cluster) may take longer to fully deprovision."
    echo -e "${GREEN}=== Cleanup Finished! ===${NC}"
    echo -e "${YELLOW}Please verify in your GCP Console that all resources have been deleted.${NC}"
}

# Run main function
main "$@"

