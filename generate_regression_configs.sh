#!/bin/bash
# Regression Configuration Generation Script
# 
# This script provides convenient commands for generating regression configurations
# and dataset info files automatically.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to show usage
show_usage() {
    echo "Regression Configuration Generation Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  all           Generate both config and dataset info files (default)"
    echo "  config        Generate only regression config files"
    echo "  dataset-info  Generate only regression dataset info files"
    echo "  clean         Clean existing files and regenerate all"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                # Generate all files"
    echo "  $0 all            # Generate all files"
    echo "  $0 config         # Generate only config files"
    echo "  $0 dataset-info   # Generate only dataset info files"
    echo "  $0 clean          # Clean and regenerate all files"
}

# Check if Python script exists
check_dependencies() {
    if [ ! -f "regression_automation_pipeline.py" ]; then
        print_error "regression_automation_pipeline.py not found!"
        print_info "Please make sure you're in the correct directory."
        exit 1
    fi
}

# Main execution
main() {
    local command="${1:-all}"
    
    case "$command" in
        "all"|"")
            print_info "Generating all regression configuration files..."
            python regression_automation_pipeline.py
            ;;
        "config")
            print_info "Generating only regression config files..."
            python regression_automation_pipeline.py --config-only
            ;;
        "dataset-info")
            print_info "Generating only regression dataset info files..."
            python regression_automation_pipeline.py --dataset-info-only
            ;;
        "clean")
            print_info "Cleaning existing files and regenerating all..."
            python regression_automation_pipeline.py --clean
            ;;
        "help"|"-h"|"--help")
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            show_usage
            exit 1
            ;;
    esac
    
    if [ $? -eq 0 ]; then
        print_success "Command completed successfully!"
    else
        print_error "Command failed!"
        exit 1
    fi
}

# Check dependencies and run
check_dependencies
main "$@"