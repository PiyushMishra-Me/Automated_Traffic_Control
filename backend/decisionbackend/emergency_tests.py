"""
emergency_tests.py
Wrapper for emergency package tests across Phases 1, 2, 3, 4A, 4B, and 5.
"""

import unittest
from backend.decisionbackend.emergency.tests import (
    TestEmergencyDecisionPhase1,
    TestEmergencyDecisionPhase2,
    TestEmergencyDecisionPhase3,
)
from backend.decisionbackend.emergency.integration_tests import (
    TestCameraIntegrationPhase4A,
    TestVisionEmergencyBridgePhase4B,
    TestEmergencyOrchestratorPhase5,
    TestUnannouncedEmergencyPhase5_1,
)

if __name__ == "__main__":
    unittest.main()
