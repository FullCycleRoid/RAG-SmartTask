"""
Script to run LangSmith RAG evaluation - FIXED VERSION
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.logger import logger
from app.evaluation.langsmith_evaluator import langsmith_evaluator


async def main():
    """Main evaluation function"""
    try:
        logger.info("Starting LangSmith RAG evaluation...")

        # Run evaluation
        results = await langsmith_evaluator.run_evaluation()

        # Calculate statistics
        stats = await langsmith_evaluator.get_evaluation_stats(results)

        # Print results
        print("\n" + "=" * 50)
        print("LANGSMITH EVALUATION RESULTS")
        print("=" * 50)

        if "error" in stats:
            print(f"Error: {stats['error']}")
        else:
            print(f"Total examples: {stats.get('total_examples', 0)}")
            print(
                f"Correctness Accuracy: {stats.get('correctness', {}).get('accuracy', 0):.2%}"
            )
            print(
                f"Relevance Accuracy: {stats.get('relevance', {}).get('accuracy', 0):.2%}"
            )
            print(f"Evaluation timestamp: {stats.get('timestamp', 'Unknown')}")

        print("=" * 50)

        # Save detailed results
        output_file = Path("evaluation_results.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Detailed results saved to: {output_file}")

        return stats

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"\n❌ Evaluation failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    result = asyncio.run(main())
    if result and "error" in result:
        sys.exit(1)
