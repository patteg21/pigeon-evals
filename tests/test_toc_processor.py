import pytest
import json
from pathlib import Path
from unittest.mock import patch, Mock

from evals.src.processor.toc_processor import TOCProcessor
from evals.src.utils.types import SECDocument


class TestTOCProcessor:
    """Test cases for TOC processor with AAPL document."""

    @pytest.fixture
    def aapl_document(self):
        """Load real AAPL document for testing."""
        doc_path = Path(__file__).parent.parent / "data/docs/AAPL/AAPL_10K_2020-10-30.txt"
        
        if not doc_path.exists():
            pytest.skip("Test document not available")
        
        with open(doc_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return SECDocument(
            ticker="AAPL",
            company="Apple Inc",
            date="2020-10-30",
            text=text,
            path=str(doc_path),
            form_type="10K"
        )

    @pytest.fixture
    def mock_toc_structure(self):
        """Mock TOC structure for AAPL."""
        return {
            "Part I": [
                "Item 1. Business",
                "Item 1A. Risk Factors", 
                "Item 1B. Unresolved Staff Comments",
                "Item 2. Properties",
                "Item 3. Legal Proceedings",
                "Item 4. Mine Safety Disclosures"
            ],
            "Part II": [
                "Item 5. Market for Registrant's Common Equity",
                "Item 6. Selected Financial Data",
                "Item 7. Management's Discussion and Analysis",
                "Item 7A. Quantitative and Qualitative Disclosures",
                "Item 8. Financial Statements and Supplementary Data",
                "Item 9. Changes in and Disagreements with Accountants",
                "Item 9A. Controls and Procedures",
                "Item 9B. Other Information"
            ],
            "Part III": [
                "Item 10. Directors, Executive Officers and Corporate Governance",
                "Item 11. Executive Compensation",
                "Item 12. Security Ownership of Certain Beneficial Owners",
                "Item 13. Certain Relationships and Related Transactions",
                "Item 14. Principal Accounting Fees and Services"
            ],
            "Part IV": [
                "Item 15. Exhibits, Financial Statement Schedules",
                "Item 16. Form 10-K Summary"
            ]
        }

    @patch('evals.src.utils.llm.OpenAILLM')
    def test_toc_processor_chunks_output(self, mock_llm_class, aapl_document, mock_toc_structure):
        """Test TOC processor and output chunks to data/tests/."""
        # Mock LLM instance and response
        mock_llm = Mock()
        mock_llm.generate_response.return_value = mock_toc_structure
        mock_llm_class.return_value = mock_llm
        
        # Create processor and process document
        processor = TOCProcessor()
        chunks = processor.process(aapl_document)
        
        # Verify we got chunks
        assert len(chunks) > 0
        print(f"Generated {len(chunks)} chunks")
        
        # Count chunk types
        chunk_types = {}
        for chunk in chunks:
            chunk_types[chunk.type_chunk] = chunk_types.get(chunk.type_chunk, 0) + 1
        
        print(f"Chunk types: {chunk_types}")
        
        # Create output directory
        output_dir = Path(__file__).parent.parent / "data/tests"
        output_dir.mkdir(exist_ok=True)
        
        # Save chunks summary
        chunks_summary = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "id": chunk.id,
                "type_chunk": chunk.type_chunk,
                "text_length": len(chunk.text),
                "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            }
            chunks_summary.append(chunk_data)
        
        summary_path = output_dir / "aapl_toc_chunks_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_summary, f, indent=2, ensure_ascii=False)
        
        # Save full chunks
        chunks_full = []
        for chunk in chunks:
            chunk_data = {
                "id": chunk.id,
                "type_chunk": chunk.type_chunk,
                "text_length": len(chunk.text),
                "text": chunk.text
            }
            chunks_full.append(chunk_data)
        
        full_path = output_dir / "aapl_toc_chunks_full.json"
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_full, f, indent=2, ensure_ascii=False)
        
        print(f"Saved chunks summary to: {summary_path}")
        print(f"Saved full chunks to: {full_path}")
        
        # Basic assertions
        assert any(c.type_chunk == "part" for c in chunks), "No part chunks found"
        assert any(c.type_chunk == "item" for c in chunks), "No item chunks found"
        
        # Print some sample chunks for verification
        for chunk in chunks[:5]:
            print(f"\nChunk type: {chunk.type_chunk}")
            print(f"Text length: {len(chunk.text)}")
            print(f"Preview: {chunk.text[:100]}...")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])