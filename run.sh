#!/bin/bash

# Set default parameters
MODEL="gpt-4"
DATASET="Loong"
STRUCTURED=true
CHUNK=false
DOCUMENT=true

# Display help information
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -m, --model MODEL       Select model (default: gpt-4o)"
    echo "  -d, --dataset DATASET   Select dataset (default: Loong)" 
    echo "  -s, --structured        Use structured processing (default: true)"
    echo "  -c, --chunk            Use chunk processing (default: false)"
    echo "  -D, --document         Use document processing (default: true)"
    echo "  -h, --help             Show help information"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -s|--structured)
            STRUCTURED=true
            shift
            ;;
        -c|--chunk)
            CHUNK=true
            shift
            ;;
        -D|--document)
            DOCUMENT=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Build command
CMD="python main.py --model $MODEL --dataset $DATASET"

if [ "$STRUCTURED" = true ]; then
    CMD="$CMD --structured"
fi

if [ "$CHUNK" = true ]; then
    CMD="$CMD --chunk"
fi

if [ "$DOCUMENT" = true ]; then
    CMD="$CMD --document"
fi

# Execute command
echo "Executing command: $CMD"
eval $CMD