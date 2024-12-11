using UnityEngine;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;
namespace AutonomousParking
{
    public class CarRayPerception : MonoBehaviour
    {
        private RayPerceptionSensorComponent3D rayPerceptionSensor;
        public int raysPerDirection = 25;
        public float maxRayDegrees = 180f;
        private List<Vector3> rayDirections;
        private string spot = "ParkingSpot";
        void Start()
        {
            rayPerceptionSensor = GetComponent<RayPerceptionSensorComponent3D>();
            if (rayPerceptionSensor == null)
            {
                Debug.LogError("Ray Perception Sensor Component is missing on the car.");
            }
            else
            {
                Debug.Log("Sensor Found on GameObject: " + gameObject.name);
                rayDirections = CalculateRayDirections();
            }
        }
        void Update()
        {
            if (rayPerceptionSensor != null && rayDirections != null)
            {
                float rayLength = rayPerceptionSensor.RayLength;
                List<string> detectableTags = rayPerceptionSensor.DetectableTags;
                foreach (var rayDirection in rayDirections)
                {
                    Vector3 worldDirection = transform.TransformDirection(rayDirection);
                    RaycastHit hit;
                    if (Physics.Raycast(transform.position, worldDirection, out hit, rayLength))
                    {
                        if (detectableTags.Contains(hit.collider.tag))
                        {
                            if (hit.collider.tag == spot)
                            {
                                GameObject spotObject = hit.collider.gameObject;
                                Debug.Log("Detected object: " + spotObject.name + " at distance: " + hit.distance);
                            }
                        }
                    }
                }
            }
        }
        public List<Vector3> CalculateRayDirections()
        {
            List<Vector3> directions = new List<Vector3>();
            float angleIncrement = (2 * maxRayDegrees) / (2 * raysPerDirection);
            int halfRays = raysPerDirection;
            for (int i = -halfRays; i <= halfRays; i++)
            {
                float angle = i * angleIncrement;
                Vector3 direction = Quaternion.Euler(0, angle, 0) * Vector3.forward;
                directions.Add(direction);
            }
            return directions;
        }
        // Add this method to return ray distances
        public List<float> GetRayDistances()
        {
            List<float> rayDistances = new List<float>();
            float rayLength = rayPerceptionSensor.RayLength;
            List<string> detectableTags = rayPerceptionSensor.DetectableTags;
            foreach (var rayDirection in rayDirections)
            {
                Vector3 worldDirection = transform.TransformDirection(rayDirection);
                RaycastHit hit;
                if (Physics.Raycast(transform.position, worldDirection, out hit, rayLength))
                {
                    if (detectableTags.Contains(hit.collider.tag))
                    {
                        rayDistances.Add(hit.distance);
                    }
                    else
                    {
                        // If no object is detected, add the max ray length
                        rayDistances.Add(rayLength);
                    }
                }
                else
                {
                    // If no hit, add the max ray length
                    rayDistances.Add(rayLength);
                }
            }
            return rayDistances;
        }
    }
}