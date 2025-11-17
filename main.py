import glob
import json
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

from src.rag_pipeline import RAGPipeline
from create_parser import create_parser

from src.imp import Datastore, Indexer, Retriever, ResponseGenerator, Evaluator


DEFAULT_SOURCE_PATH = "sample_data/source/"
DEFAULT_EVAL_PATH = "sample_data/eval/level-1.json"


def create_pipeline() -> RAGPipeline:
    """Create and return a new RAG Pipeline instance with all components."""
    datastore = Datastore()
    indexer = Indexer()
    retriever = Retriever(datastore=datastore)
    response_generator = ResponseGenerator()
    evaluator = Evaluator()
    return RAGPipeline(datastore, indexer, retriever, response_generator, evaluator)


def main():
    parser = create_parser()  # Create the CLI parser
    args = parser.parse_args()
    pipeline = create_pipeline()

    # Process source paths and eval path
    source_path = args.path if args.path else DEFAULT_SOURCE_PATH
    eval_path = args.eval_file if args.eval_file else DEFAULT_EVAL_PATH
    document_paths = get_files_in_directory(source_path)

    # Execute commands
    if args.command in ["reset", "run"]:
        print("🗑️  Resetting the database...")
        pipeline.reset()

    if args.command in ["add", "run"]:
        print(f"🔍 Adding documents: {', '.join(document_paths)}")
        pipeline.add_documents(document_paths)

    if args.command in ["evaluate", "run"]:
        print(f"📊 Evaluating using questions from: {eval_path}")
        with open(eval_path, "r") as file:
            sample_questions = json.load(file)
        # Handle --start parameter (1-indexed)
        start_idx = 0
        if hasattr(args, 'start') and args.start:
            start_idx = args.start - 1  # Convert to 0-indexed
            if start_idx < 0 or start_idx >= len(sample_questions):
                print(f"❌ Invalid start index. Must be between 1 and {len(sample_questions)}")
                return
            sample_questions = sample_questions[start_idx:]
            print(f"🔬 Starting from question #{args.start}")
        # Limit questions if --limit/-n is specified
        if hasattr(args, 'limit') and args.limit:
            sample_questions = sample_questions[:args.limit]
            print(f"🔬 Testing {args.limit} question(s)")
        pipeline.evaluate(sample_questions)

    if args.command == "query":
        print(f"✨ Response: {pipeline.process_query(args.prompt)}")


def get_files_in_directory(source_path: str) -> List[str]:
    if os.path.isfile(source_path):
        return [source_path]
    return glob.glob(os.path.join(source_path, "*"))


if __name__ == "__main__":
    main()
