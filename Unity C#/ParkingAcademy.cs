using UnityEngine;
using Unity.MLAgents;

namespace AutonomousParking
{
    public class ParkingAcademy : MonoBehaviour
    {
        // Example global parameter
        public float maxSteps = 500f;

        private void Start()
        {
            // Initialize any global settings if necessary
            Debug.Log("Academy Initialized");
        }

        private void Update()
        {
            // Perform any environment-wide updates if needed
        }

        public void ResetEnvironment()
        {
            // This method can be called to reset the entire environment
            Debug.Log("Environment Reset");
        }
    }
}




