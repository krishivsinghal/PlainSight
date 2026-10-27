import os
import json
import time
import hashlib
import re
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
from prompts.extraction_prompt import EXTRACTION_PROMPT
from prompts.summary_prompt import SUMMARY_PROMPT
from prompts.risk_prompt import RISK_PROMPT
from dotenv import load_dotenv
import streamlit as st
import logging
from pathlib import Path

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Mock data for development without API calls
MOCK_MODE = os.getenv("MOCK_MODE", "False").lower() == "true"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_legal_context(clause_text: str) -> str:
    """Retrieve optional legal context for the clause-risk prompt."""
    hybrid_dir = os.getenv("LEGAL_HYBRID_INDEX_DIR")
    vector_dir = os.getenv("LEGAL_VECTOR_STORE_DIR")
    try:
        if hybrid_dir:
            from retrieval.hybrid_index import hybrid_search

            matches = hybrid_search(clause_text, topk=3, index_dir=Path(hybrid_dir))
            return "\n\n".join(
                f"Source: {chunk['meta'].get('citation') or chunk['meta'].get('case') or chunk['meta'].get('file')}\n{chunk['text']}"
                for _, chunk in matches
            )
        if vector_dir:
            from retrieval.vector_store import LegalVectorStore

            return LegalVectorStore(persist_directory=vector_dir).get_legal_context(clause_text)
    except Exception as exc:
        logger.warning("Legal retrieval unavailable: %s", exc)
    return ""

# Cache the extraction function
@st.cache_data(ttl=3600, show_spinner=False)
def extract_clauses(document_text: str) -> Dict[str, str]:
    """
    Extract clauses from contract text using GPT
    
    Args:
        document_text: Full text of the contract document
        
    Returns:
        Dictionary with clause titles as keys and clause text as values
    """
    if MOCK_MODE:
        # Return mock data for development
        return {
            "1. Definitions": "In this Agreement: 'Service' means the consulting services...",
            "2. Scope of Work": "The Consultant shall provide the following services to the Client...",
            "3. Payment Terms": "The Client shall pay the Consultant within 30 days of receipt of invoice...",
            "4. Intellectual Property": "All intellectual property created during the provision of services...",
            "5. Confidentiality": "Each party shall maintain the confidentiality of all information...",
            "6. Termination": "This Agreement may be terminated by either party with 30 days notice...",
            "7. Limitation of Liability": "The Consultant's liability shall not exceed the fees paid...",
            "8. Governing Law": "This Agreement is governed by the laws of New South Wales..."
        }
    
    # Log that we're starting clause extraction
    logger.info("Starting clause extraction with LLM (EXTRACTION)")
    
    # Prepare prompt with the document text
    prompt = EXTRACTION_PROMPT.format(document_text=document_text)
    logger.info(f"Extraction Prompt: {prompt}")

    try:
        # Call OpenAI API
        response = client.chat.completions.create(model="gpt-4o-mini",  # or "gpt-3.5-turbo" for lower cost
            messages=[
                {"role": "system", "content": "You are a legal assistant specializing in Australian contract law. Your task is to extract and identify distinct clauses from contracts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for more consistent results
            max_tokens=10000
        )

        logger.info(f"Response from LLM in extraction: {response}")
        
        # Parse the response
        try:
            clauses_json = response.choices[0].message.content
            # Extract JSON if it's within markdown code blocks
            if "```json" in clauses_json:
                clauses_json = clauses_json.split("```json")[1].split("```")[0]
            elif "```" in clauses_json:
                clauses_json = clauses_json.split("```")[1].split("```")[0]
            
            # Parse the JSON response
            clauses = json.loads(clauses_json)
            
            # Log the LLM response for debugging
            logger.info(f"LLM provided a valid response and we were able to parse it")
            
            return clauses
        except json.JSONDecodeError:
            # Fallback to regex-based extraction if JSON parsing fails
            from utils.document_parser import identify_clauses_regex
            logger.info(f"LLM did not provide a response. We fallback to regex-based extraction")
            return identify_clauses_regex(document_text)
            
    except Exception as e:
        print(f"Error in GPT clause extraction: {str(e)}")
        logger.info(f"LLM did not provide a response. There was an error in the clause extraction.")
        # Fallback to regex-based extraction
        from utils.document_parser import identify_clauses_regex
        return identify_clauses_regex(document_text)

# Cache the summarization function
@st.cache_data(ttl=3600, show_spinner=False)
def summarize_clause(clause_title: str, clause_text: str) -> str:
    """
    Generate a plain English summary of a contract clause
    
    Args:
        clause_title: Title or identifier of the clause
        clause_text: Full text of the clause
        
    Returns:
        Plain English summary of the clause
    """
    logger.info("Starting summarization with LLM (SUMMARIZATION)")
    if MOCK_MODE:
        # Return mock data for development
        mock_summaries = {
            "1. Definitions": "This section defines key terms used throughout the agreement, including what constitutes the 'Service' and who the parties are.",
            "2. Scope of Work": "This outlines exactly what work the consultant will do, including deliverables and timelines.",
            "3. Payment Terms": "You must pay within 30 days of receiving an invoice. Late payments may incur additional fees.",
            "4. Intellectual Property": "Any work created during the project belongs to the client after payment is complete.",
            "5. Confidentiality": "Both parties must keep sensitive business information private and not share it with others.",
            "6. Termination": "Either party can end the agreement with 30 days written notice. Immediate termination is possible if there's a serious breach.",
            "7. Limitation of Liability": "The consultant won't be responsible for damages beyond the amount you've paid them.",
            "8. Governing Law": "If there's a dispute, New South Wales law applies and any legal proceedings must happen in NSW courts."
        }
        
        # Return mock summary if available, otherwise generate a generic one
        return mock_summaries.get(clause_title, f"This clause covers {clause_title.lower()} terms.")
    
    # Prepare prompt
    prompt = SUMMARY_PROMPT.format(clause_title=clause_title, clause_text=clause_text)
    
    try:
        # Call OpenAI API
        response = client.chat.completions.create(model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a legal assistant specializing in explaining Australian contract law in plain English. Your goal is to make complex legal language understandable for small business owners."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,  # Slightly higher for more natural language
            max_tokens=500
        )
        
        # Return the summary
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error in GPT summarization: {str(e)}")
        # Return a generic summary if API call fails
        return f"This clause appears to address {clause_title.lower()}. Please review the original text for details."

# Cache the risk analysis function
@st.cache_data(ttl=3600, show_spinner=False)
def analyze_risks(clause_title: str, clause_text: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Analyze a clause for potential legal risks based on Australian law
    
    Args:
        clause_title: Title or identifier of the clause
        clause_text: Full text of the clause
        
    Returns:
        Tuple containing:
        - List of risk statements (for backward compatibility)
        - List of dictionaries with detailed risk information including problematic text
    """
    logger.info("Starting risk analysis with LLM (RISK ANALYSIS)")
    if MOCK_MODE:
        # Return mock data for development
        mock_risks = {
            "1. Definitions": [],
            "2. Scope of Work": [],
            "3. Payment Terms": [
                {
                    "problematic_text": "The Client shall pay the Consultant within 30 days of receipt of invoice",
                    "explanation": "The 30-day payment term may be too long for small businesses with cash flow concerns.",
                    "legal_reference": "ACCC guidelines on fair payment terms for small businesses",
                    "severity": "medium"
                }
            ],
            "4. Intellectual Property": [],
            "5. Confidentiality": [
                {
                    "problematic_text": "Each party shall maintain the confidentiality of all information",
                    "explanation": "The confidentiality obligations continue indefinitely, which may be overly restrictive.",
                    "legal_reference": "Australian common law on restraint of trade",
                    "severity": "medium"
                }
            ],
            "6. Termination": [
                {
                    "problematic_text": "This Agreement may be terminated by either party with 30 days notice",
                    "explanation": "The 30-day notice period for termination may be problematic if you need to exit quickly.",
                    "legal_reference": "ACCC guidelines on fair termination clauses",
                    "severity": "low"
                }
            ],
            "7. Limitation of Liability": [
                {
                    "problematic_text": "The Consultant's liability shall not exceed the fees paid",
                    "explanation": "This broad limitation of liability clause may be unenforceable under Australian Consumer Law for certain types of loss.",
                    "legal_reference": "Section 64A of the Australian Consumer Law",
                    "severity": "high"
                }
            ],
            "8. Governing Law": []
        }
        
        # Get the detailed risk information
        detailed_risks = mock_risks.get(clause_title, [])
        
        # For backward compatibility, also return the simple risk statements
        simple_risks = [risk["explanation"] for risk in detailed_risks]
        
        return simple_risks, detailed_risks
    
    # Prepare prompt, with retrieved context when an index is configured.
    prompt = RISK_PROMPT.format(clause_title=clause_title, clause_text=clause_text)
    legal_context = get_legal_context(clause_text)
    if legal_context:
        prompt += "\n\nRetrieved legal context (check source relevance):\n" + legal_context
    
    try:
        # Call OpenAI API
        response = client.chat.completions.create(model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a legal risk analysis system specializing in Australian contract law. Identify potential risks in contract clauses for small businesses, focusing on unfair contract terms under the Australian Consumer Law and ACCC guidelines. You must respond with valid JSON in the exact format specified in the prompt."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=2000
        )
        
        # Parse the response
        response_text = response.choices[0].message.content.strip()
        logger.info("=== DEBUG: Risk Response from OpenAI ===")
        logger.info(response_text)
        logger.info("=== END DEBUG ===")
        
        # Try to parse as JSON
        detailed_risks = []
        simple_risks = []
        
        try:
            # First try to find JSON in the response
            json_str = None
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0]
                logger.info("=== DEBUG: Found JSON in ```json block ===")
                logger.info(json_str)
                logger.info("=== END DEBUG ===")
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0]
                logger.info("=== DEBUG: Found JSON in ``` block ===")
                logger.info(json_str)
                logger.info("=== END DEBUG ===")
            else:
                # Try to find JSON array in the text
                json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    st.write("=== DEBUG: Found JSON array in text ===")
                    st.write(json_str)
                    st.write("=== END DEBUG ===")
                else:
                    json_str = response_text
                    st.write("=== DEBUG: Using entire response as JSON ===")
                    st.write(json_str)
                    st.write("=== END DEBUG ===")
            
            # Clean up the JSON string
            json_str = json_str.strip()
            # Remove any trailing commas
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            
            # Parse the JSON
            detailed_risks = json.loads(json_str)
            
            # Validate the structure
            if not isinstance(detailed_risks, list):
                raise ValueError("Response is not a JSON array")
            
            # Extract simple risks for backward compatibility
            for risk_item in detailed_risks:
                if not isinstance(risk_item, dict):
                    raise ValueError("Risk item is not a dictionary")
                if "explanation" not in risk_item:
                    raise ValueError("Risk item missing 'explanation' field")
                simple_risks.append(risk_item.get("explanation", ""))
                
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            logger.info("=== ERROR: JSON parsing failed in Risk  ===")
            logger.info(f"Error type: {type(e).__name__}")
            logger.info(f"Error message: {str(e)}")
            logger.info(f"Response text: {response_text}")
            logger.info(f"JSON string that failed: {json_str if 'json_str' in locals() else 'Not found'}")
            logger.info("=== END ERROR ===")
            
            # Fallback to the old method for backward compatibility
            if "no significant risks" in response_text.lower() or "no risks" in response_text.lower():
                return [], []
            
            # Extract risk points
            for line in response_text.split('\n'):
                line = line.strip()
                # Look for list items or paragraphs that describe risks
                if (line.startswith('- ') or line.startswith('• ') or 
                    line.startswith('* ') or line.startswith('Risk:')):
                    # Clean up the line
                    risk = line.lstrip('- •*').strip()
                    if risk.lower().startswith('risk:'):
                        risk = risk[5:].strip()
                    
                    if risk and len(risk) > 10:  # Only include substantial risk descriptions
                        # Create a proper dictionary structure for the risk
                        risk_dict = {
                            "problematic_text": "",  # We don't have the exact text in this case
                            "explanation": risk,
                            "legal_reference": "General Australian contract law principles",
                            "severity": "medium"  # Default to medium severity
                        }
                        detailed_risks.append(risk_dict)
                        simple_risks.append(risk)
            
            # If we couldn't parse list items but there's content, use the whole response
            if not detailed_risks and len(response_text) > 10:
                risk_dict = {
                    "problematic_text": "",  # We don't have the exact text in this case
                    "explanation": response_text,
                    "legal_reference": "General Australian contract law principles",
                    "severity": "medium"  # Default to medium severity
                }
                detailed_risks = [risk_dict]
                simple_risks = [response_text]
                
        return simple_risks, detailed_risks
    
    except Exception as e:
        print(f"Error in GPT risk analysis: {str(e)}")
        logger.info(f"Error in GPT risk analysis: {str(e)}")
        # Return empty lists if API call fails
        return [], []
