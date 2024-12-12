from typing import Dict, Any
import numpy as np

class ResultsFormatter:
    # Reference ranges based on typical gestational age (example values)
    FETAL_MEASUREMENTS_LABELS = {
        'CRL': {'name': 'Crown-Rump Length', 'unit': 'mm'},
        'HC': {'name': 'Head Circumference', 'unit': 'mm'},
        'AC': {'name': 'Abdominal Circumference', 'unit': 'mm'},
        'FL': {'name': 'Femur Length', 'unit': 'mm'}
    }
    
    ANOMALY_TYPES = {
        0: "Neural Tube Defect",
        1: "Heart Abnormality",
        2: "Abdominal Wall Defect",
        3: "Renal Anomaly",
        4: "Skeletal Dysplasia",
        5: "Facial Cleft",
        6: "Growth Restriction",
        7: "Placental Abnormality",
        8: "Amniotic Fluid Issue",
        9: "Other Structural Anomaly"
    }

    @staticmethod
    def format_predictions(raw_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Format raw predictions into human-readable results while preserving raw data."""
        
        formatted_results = {
            "summary": {},
            "detailed_measurements": {},
            "risk_assessment": {},
            "recommendations": [],
            "raw_data": raw_predictions  # Include the raw predictions
        }

        # Format fetal measurements
        measurements = raw_predictions['fetal_measurement']
        formatted_results["detailed_measurements"]["fetal_measurements"] = {
            label: {
                "name": info["name"],
                "value": float(measurements[i]),
                "unit": info["unit"]
            }
            for i, (label, info) in enumerate(ResultsFormatter.FETAL_MEASUREMENTS_LABELS.items())
        }

        # Format health assessment
        health_probs = raw_predictions['health_assessment']
        health_status = "Normal" if health_probs[0] > health_probs[1] else "Requires Attention"
        confidence = max(health_probs) * 100
        
        formatted_results["summary"]["health_status"] = {
            "status": health_status,
            "confidence": f"{confidence:.1f}%"
        }

        # Format gender prediction
        gender_probs = raw_predictions['gender']
        predicted_gender = "Female" if gender_probs[1] > gender_probs[0] else "Male"
        gender_confidence = max(gender_probs) * 100
        
        formatted_results["summary"]["gender_prediction"] = {
            "prediction": predicted_gender,
            "confidence": f"{gender_confidence:.1f}%"
        }

        # Format anomaly detection
        anomaly_probs = raw_predictions['anomaly']
        significant_anomalies = []
        
        for i, prob in enumerate(anomaly_probs):
            if prob > 0.3:  # Threshold for reporting
                significant_anomalies.append({
                    "type": ResultsFormatter.ANOMALY_TYPES[i],
                    "probability": f"{prob * 100:.1f}%"
                })
        
        formatted_results["risk_assessment"]["potential_concerns"] = significant_anomalies

        # Add recommendations based on findings
        formatted_results["recommendations"] = ResultsFormatter._generate_recommendations(
            health_status,
            significant_anomalies
        )

        return formatted_results

    @staticmethod
    def _generate_recommendations(health_status: str, anomalies: list) -> list:
        """Generate recommendations based on findings."""
        recommendations = []
        
        if health_status == "Requires Attention":
            recommendations.append(
                "Schedule a follow-up consultation with your healthcare provider."
            )
        
        if len(anomalies) > 0:
            recommendations.append(
                "Additional detailed scanning may be recommended for specific areas of interest."
            )
        
        if not recommendations:
            recommendations.append(
                "Continue with regular prenatal check-ups as scheduled."
            )

        return recommendations