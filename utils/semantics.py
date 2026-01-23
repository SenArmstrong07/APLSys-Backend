import re
from typing import List, Dict, Any, Tuple

# Validation patterns and confidence rules
VALIDATION_RULES = {
    "degree": {
        "patterns": [r"\b(Bachelor|BS|BA|Master|MS|MA|PhD|Associate|Diploma|Certificate|BBA|BSC|MSC|MBA|B\.S\.|M\.S\.|B\.A\.|M\.A\.)\b"],
        "required_keywords": ["bachelor", "master", "degree", "bs", "ba", "ms", "ma", "phd", "associate", "diploma", "certificate"],
        "forbidden_keywords": ["height", "weight", "age", "gender", "location", "phone"],
        "min_length": 3,
    },
    "school": {
        "patterns": [r"(?:University|College|Institute|Academy|School|State)"],
        "min_length": 5,
        "forbidden_keywords": ["height", "weight", "years"]
    },
    "company": {
        "patterns": [r"(?:Inc|Corp|LLC|Ltd|Company|Group|Services|Solutions)"],
        "min_length": 3,
    },
    "years": {
        "pattern": r"(19[5-9]\d|20[0-3]\d)",
        "length": 4,
    },
    "email": {
        "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "strict": True
    },
    "phone": {
        "pattern": r"^\+?[1-9]\d{1,14}$",
    }
}

class ConfidenceScorer:
    """Calculate confidence scores for extracted fields."""
    
    @staticmethod
    def score_degree(text: str) -> Tuple[float, str]:
        """Score a degree field (0-100)."""
        if not text or len(text) < 3:
            return 0.0, "too_short"
        
        score = 0.0
        reasons = []
        
        text_lower = text.lower()
        
        # Check for required keywords
        required = VALIDATION_RULES["degree"]["required_keywords"]
        keyword_match = sum(1 for kw in required if kw in text_lower)
        if keyword_match > 0:
            score += 40
            reasons.append(f"keyword_match({keyword_match})")
        
        # Check for forbidden keywords
        forbidden = VALIDATION_RULES["degree"]["forbidden_keywords"]
        if any(fw in text_lower for fw in forbidden):
            score -= 30
            reasons.append("contains_forbidden_keyword")
        
        # Check length (longer = better, but suspicious if too long)
        if 10 <= len(text) <= 80:
            score += 20
            reasons.append("length_ok")
        elif len(text) > 80:
            score -= 10
            reasons.append("too_long")
        
        # Check for pattern match
        if re.search(VALIDATION_RULES["degree"]["patterns"][0], text, re.IGNORECASE):
            score += 40
            reasons.append("pattern_match")
        
        # Normalize score
        score = max(0, min(100, score))
        return score, ", ".join(reasons)
    
    @staticmethod
    def score_school(text: str) -> Tuple[float, str]:
        """Score a school/university field (0-100)."""
        if not text or len(text) < 5:
            return 0.0, "too_short"
        
        score = 20.0  # Base score for any school mention
        reasons = ["base_score"]
        
        text_lower = text.lower()
        
        # Check for institution keywords
        if re.search(r"university|college|institute|academy|school", text_lower):
            score += 50
            reasons.append("institution_keyword_found")
        
        # Check for forbidden keywords
        forbidden = VALIDATION_RULES["school"]["forbidden_keywords"]
        if any(fw in text_lower for fw in forbidden):
            score -= 30
            reasons.append("forbidden_keyword")
        
        # Length check
        if 5 <= len(text) <= 100:
            score += 30
            reasons.append("length_ok")
        
        score = max(0, min(100, score))
        return score, ", ".join(reasons)
    
    @staticmethod
    def score_year(text: str) -> Tuple[float, str]:
        """Score a year field (0-100)."""
        match = re.search(VALIDATION_RULES["years"]["pattern"], text)
        if not match:
            return 0.0, "no_valid_year"
        
        year = int(match.group(0))
        if 1950 <= year <= 2035:
            return 100.0, f"valid_year({year})"
        else:
            return 0.0, f"year_out_of_range({year})"
    
    @staticmethod
    def score_email(text: str) -> Tuple[float, str]:
        """Score an email field (0-100)."""
        if not text:
            return 0.0, "empty"
        
        # Strict email validation
        if re.match(VALIDATION_RULES["email"]["pattern"], text):
            # Additional checks
            if text.count("@") == 1 and "." in text.split("@")[1]:
                return 100.0, "valid_email"
        
        return 0.0, "invalid_email_format"
    
    @staticmethod
    def score_phone(text: str) -> Tuple[float, str]:
        """Score a phone field (0-100)."""
        if not text:
            return 0.0, "empty"
        
        # Remove common formatting characters
        clean = re.sub(r"[\s\-\(\)\.]+", "", text)
        
        if re.match(VALIDATION_RULES["phone"]["pattern"], clean):
            return 100.0, "valid_phone"
        
        return 0.0, "invalid_phone_format"


class EducationExtractor:
    """Extract structured education entries from text."""
    
    degree_keywords = ["bachelor", "master", "phd", "associate", "diploma", "certificate", "bs", "ba", "ms", "ma", "bba", "bsc", "msc", "mba"]
    school_keywords = ["university", "college", "institute", "academy", "school"]
    
    @staticmethod
    def extract_education_entries(text: str) -> List[Dict[str, Any]]:
        """
        Extract education entries from education section text.
        Returns list of structured education dicts with confidence scores.
        """
        if not text or len(text.strip()) < 10:
            return []
        
        entries = []
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        
        current_entry = {
            "school": "",
            "degree": "",
            "years": [],
            "details": "",
            "confidence": 0.0,
        }
        
        for line in lines:
            # Try to extract school
            if re.search(r"(?:university|college|institute|academy|school)", line, re.IGNORECASE):
                if current_entry["school"]:
                    # Save previous entry
                    entries.append(EducationExtractor._finalize_entry(current_entry))
                    current_entry = {"school": "", "degree": "", "years": [], "details": "", "confidence": 0.0}
                
                current_entry["school"] = line
            
            # Try to extract degree
            elif re.search(r"(?:bachelor|master|phd|associate|diploma|certificate|b\.s\.|m\.s\.|b\.a\.|m\.a\.|bs|ba|ms|ma|mba|bba)", line, re.IGNORECASE):
                current_entry["degree"] = line
            
            # Try to extract years
            elif re.search(r"\b(19[5-9]\d|20[0-3]\d)\b", line):
                years = re.findall(r"\b(19[5-9]\d|20[0-3]\d)\b", line)
                current_entry["years"].extend(years)
            
            # Otherwise, accumulate as details
            else:
                if line:
                    current_entry["details"] += line + " "
        
        # Finalize last entry
        if current_entry["school"] or current_entry["degree"]:
            entries.append(EducationExtractor._finalize_entry(current_entry))
        
        return entries
    
    @staticmethod
    def _finalize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize and validate an education entry."""
        entry["details"] = entry["details"].strip()
        
        # Calculate confidence scores
        school_score, school_reason = ConfidenceScorer.score_school(entry["school"]) if entry["school"] else (0.0, "empty")
        degree_score, degree_reason = ConfidenceScorer.score_degree(entry["degree"]) if entry["degree"] else (0.0, "empty")
        
        # Aggregate confidence (weighted average)
        confidence = 0.0
        weight_sum = 0.0
        
        if entry["school"]:
            confidence += school_score * 0.4
            weight_sum += 0.4
        
        if entry["degree"]:
            confidence += degree_score * 0.5
            weight_sum += 0.5
        
        if entry["years"]:
            confidence += 100.0 * 0.1
            weight_sum += 0.1
        
        entry["confidence"] = confidence / weight_sum if weight_sum > 0 else 0.0
        entry["validation"] = {
            "school": {"score": school_score, "reason": school_reason},
            "degree": {"score": degree_score, "reason": degree_reason},
        }
        
        return entry


class ExperienceExtractor:
    """Extract structured work experience entries from text."""
    
    role_keywords = ["manager", "engineer", "developer", "analyst", "consultant", "director", "specialist", "coordinator", "lead"]
    
    @staticmethod
    def extract_experience_entries(text: str) -> List[Dict[str, Any]]:
        """
        Extract work experience entries from experience section text.
        Returns list of structured experience dicts with confidence scores.
        """
        if not text or len(text.strip()) < 10:
            return []
        
        entries = []
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        
        current_entry = {
            "company": "",
            "role": "",
            "years": [],
            "descriptions": [],
            "confidence": 0.0,
        }
        
        for line in lines:
            # Check if this looks like a company line (all caps or contains company keywords)
            if (line.isupper() or re.search(r"(?:Inc|Corp|LLC|Ltd|Company|Group|Services|Solutions)", line)) and len(line) < 100:
                if current_entry["company"] or current_entry["role"]:
                    # Save previous entry
                    entries.append(ExperienceExtractor._finalize_entry(current_entry))
                    current_entry = {"company": "", "role": "", "years": [], "descriptions": [], "confidence": 0.0}
                
                current_entry["company"] = line
            
            # Check if this looks like a role line
            elif re.search(r"(?:manager|engineer|developer|analyst|consultant|director|specialist|coordinator|lead)", line, re.IGNORECASE):
                current_entry["role"] = line
            
            # Try to extract years
            elif re.search(r"\b(19[5-9]\d|20[0-3]\d)\b", line):
                years = re.findall(r"\b(19[5-9]\d|20[0-3]\d)\b", line)
                current_entry["years"].extend(years)
            
            # Otherwise, accumulate as descriptions
            else:
                if line and len(line) > 5:
                    current_entry["descriptions"].append(line)
        
        # Finalize last entry
        if current_entry["company"] or current_entry["role"]:
            entries.append(ExperienceExtractor._finalize_entry(current_entry))
        
        return entries
    
    @staticmethod
    def _finalize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize and validate a work experience entry."""
        # Calculate confidence scores
        company_score = 100.0 if entry["company"] else 0.0
        role_score = 100.0 if entry["role"] else 0.0
        year_score = 100.0 if entry["years"] else 0.0
        
        # Aggregate confidence (weighted average)
        confidence = 0.0
        weight_sum = 0.0
        
        if entry["company"]:
            confidence += company_score * 0.35
            weight_sum += 0.35
        
        if entry["role"]:
            confidence += role_score * 0.40
            weight_sum += 0.40
        
        if entry["years"]:
            confidence += year_score * 0.25
            weight_sum += 0.25
        
        entry["confidence"] = confidence / weight_sum if weight_sum > 0 else 0.0
        entry["validation"] = {
            "company": {"score": company_score},
            "role": {"score": role_score},
            "years": {"score": year_score},
        }
        
        return entry