using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;

namespace AutonomousParking
{
    public class CarAgent : Agent
    {
        [SerializeField] private Transform carSpawnPoint;
        [SerializeField] private Rigidbody carRigidbody;
        [SerializeField] private WheelCollider frontLeftWheel, frontRightWheel, rearLeftWheel, rearRightWheel;
        [SerializeField] private CarRayPerception rayPerceptionSensor;

        private Transform parkingSpot; // Dynamically assigned parking spot
        private const float MaxDistance = 20f; // Max distance for normalization
        private float collisionPenalty; // Tracks penalty due to collisions
        private float stationaryTime = 0f; // Tracks how long the car has been stationary
        private const float StationaryThreshold = 0.5f; // Velocity below this threshold is considered stationary
        private const float StationaryPenaltyTime = 2f; // Time threshold to start penalizing stationary behavior
        private const float StationaryPenalty = -0.05f; // Penalty for staying stationary
        private float previousDistanceToTarget = float.MaxValue; // Tracks distance to target (entrance or parking spot)
        private float previousParkingSpotDistance = float.MaxValue; // Tracks distance to parking spot

        private bool reachedEntrance = false; // Flag to track if the entrance is reached
        [SerializeField] private Transform entrance; // Entrance collider transform

        // Constants for obstacle avoidance
        private const float ObstacleProximityThreshold = 2f; // Distance threshold for obstacle proximity
        private const float MaxObstaclePenalty = -0.5f; // Maximum penalty for very close obstacles

        public override void OnEpisodeBegin()
        {
            // Define the bounds of the rectangular area
            float minX = -6.36f;
            float maxX = 2.87f;
            float minZ = -27.49f;
            float maxZ = -20.29f;

            // Generate a random position within the rectangle
            float randomX = Random.Range(minX, maxX);
            float randomZ = Random.Range(minZ, maxZ);
            Vector3 randomPosition = new Vector3(randomX, carSpawnPoint.position.y, randomZ);

            // Generate a random rotation (only on the Y-axis to align with ground)
            Quaternion randomRotation = Quaternion.Euler(0, Random.Range(0, 360), 0);

            // Set the car's position and rotation
            transform.position = randomPosition;
            transform.rotation = randomRotation;

            // Reset velocity
            carRigidbody.velocity = Vector3.zero;
            carRigidbody.angularVelocity = Vector3.zero;

            // Reset WheelColliders
            ResetWheelColliders();

            // Reset rewards, penalties, and flags
            collisionPenalty = 0f;
            SetReward(0f); // Explicitly reset the total reward
            reachedEntrance = false; // Reset entrance flag
            previousDistanceToTarget = float.MaxValue; // Reset distance tracker
            previousParkingSpotDistance = float.MaxValue; // Reset parking spot distance tracker

            // Now we need to fetch the parking spot from Spawner.cs
            parkingSpot = GetRandomParkingSpot(); // Fetch the random parking spot
        }

        // Fetch the random parking spot from Spawner's list of parking spots
        private Transform GetRandomParkingSpot()
        {
            Spawner spawner = FindObjectOfType<Spawner>();
            return spawner.GetRandomParkingSpot();
        }

        // Method to calculate the pre-parking position (in front of the parking spot)
        private Vector3 GetPreParkingPosition()
        {
            // Assuming pre-parking is 3.4 meters in front of the parking spot
            Vector3 direction = (parkingSpot.position - transform.position).normalized;
            return parkingSpot.position - direction * 3.4f; // Adjust the distance as needed
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            // Add ray perception distances as observations
            if (rayPerceptionSensor != null)
            {
                List<float> rayDistances = rayPerceptionSensor.GetRayDistances();
                foreach (float distance in rayDistances)
                {
                    sensor.AddObservation(distance / rayPerceptionSensor.GetComponent<RayPerceptionSensorComponent3D>().RayLength);
                }
            }

            // Normalize Local Position (X-axis)
            // Assuming a reasonable range for X, adjust min and max values as needed
            float normalizedX = Mathf.InverseLerp(-10f, 10f, transform.localPosition.x);
            sensor.AddObservation(normalizedX);

            // Normalize Local Position (Z-axis)
            // Adjust min and max values as needed
            float normalizedZ = Mathf.InverseLerp(-30f, 0f, transform.localPosition.z);
            sensor.AddObservation(normalizedZ);

            // Normalize Yaw Angle (0-360 degrees)
            float normalizedYaw = transform.localRotation.eulerAngles.y / 360f;
            sensor.AddObservation(normalizedYaw);

            // Normalize Speed
            // Assuming max speed is around 10 m/s, adjust as needed
            float normalizedSpeed = Mathf.Clamp01(carRigidbody.velocity.magnitude / 10f);
            sensor.AddObservation(normalizedSpeed);

            // Normalize Steer Angle
            // Assuming max steer angle is 25 degrees (from ApplySteering method)
            float currentSteerAngle = (frontLeftWheel.steerAngle + frontRightWheel.steerAngle) / 2f;
            float normalizedSteerAngle = Mathf.InverseLerp(-25f, 25f, currentSteerAngle);
            sensor.AddObservation(normalizedSteerAngle);

            // Set the target: if the entrance is reached, target the pre-parking position first
            Transform target = reachedEntrance ? (Vector3.Distance(transform.position, GetPreParkingPosition()) > 1f ? parkingSpot : null) : entrance;
            Vector3 relativePosition = (target != null ? target.position : Vector3.zero) - transform.position;
            sensor.AddObservation(relativePosition / 10f); // Relative position to target
            sensor.AddObservation(relativePosition.magnitude / MaxDistance); // Distance to target
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            // Extract actions
            float throttle = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
            float steering = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

            // Apply vehicle controls
            ApplyThrottle(throttle);
            ApplySteering(steering);

            // Reset reward accumulator
            float totalReward = 0f;

            // Choose the target based on the car's state (pre-parking or parking spot)
            Transform target = reachedEntrance
                ? (Vector3.Distance(transform.position, GetPreParkingPosition()) > 1f ? parkingSpot : null)
                : entrance;

            Vector3 relativePosition = (target != null ? target.position : Vector3.zero) - transform.position;
            float distanceToTarget = relativePosition.magnitude;

            // Distance-based reward (controlled)
            if (distanceToTarget < previousDistanceToTarget)
            {
                // Reward for getting closer, but with a small, controlled increment
                float distanceReward = Mathf.Clamp((previousDistanceToTarget - distanceToTarget) * 0.05f, -0.1f, 0.1f);
                totalReward += distanceReward;
            }
            else
            {
                // Small penalty for moving away
                totalReward -= 0.02f;
            }

            // Alignment reward (controlled)
            float alignmentAngle = Vector3.Angle(transform.forward, relativePosition);
            float alignmentReward = Mathf.Clamp(1f - (alignmentAngle / 180f), -0.1f, 0.1f);
            totalReward += alignmentReward;

            // Smooth driving reward
            if (Mathf.Abs(steering) < 0.2f && Mathf.Abs(throttle) > 0.3f)
            {
                totalReward += 0.05f; // Slightly increased from previous version
            }

            // Velocity and stationary behavior management
            float velocityMagnitude = carRigidbody.velocity.magnitude;
            if (velocityMagnitude < StationaryThreshold)
            {
                stationaryTime += Time.deltaTime;
                if (stationaryTime > StationaryPenaltyTime)
                {
                    totalReward += StationaryPenalty; // Penalize extended stationary behavior
                }
            }
            else
            {
                stationaryTime = 0f; // Reset stationary time if moving
            }

            // Obstacle and Ray Perception Rewards
            if (rayPerceptionSensor != null)
            {
                List<float> rayDistances = rayPerceptionSensor.GetRayDistances();
                float closestObstacleDistance = float.MaxValue;

                // Check ray distances for obstacles
                for (int i = 0; i < rayDistances.Count; i++)
                {
                    RaycastHit hit;
                    Vector3 rayDirection = rayPerceptionSensor.transform.TransformDirection(rayPerceptionSensor.CalculateRayDirections()[i]);
                    float rayLength = rayPerceptionSensor.GetComponent<RayPerceptionSensorComponent3D>().RayLength;

                    if (Physics.Raycast(transform.position, rayDirection, out hit, rayLength))
                    {
                        // Track closest obstacle
                        if ((hit.collider.CompareTag("Wall") || hit.collider.CompareTag("Car")) && hit.distance < closestObstacleDistance)
                        {
                            closestObstacleDistance = hit.distance;
                        }
                    }
                }

                // Obstacle proximity penalty
                if (closestObstacleDistance < ObstacleProximityThreshold)
                {
                    float proximityFactor = 1f - (closestObstacleDistance / ObstacleProximityThreshold);
                    float obstaclePenalty = Mathf.Lerp(0f, MaxObstaclePenalty, proximityFactor);
                    totalReward += obstaclePenalty;
                }
            }

            // Add collision penalty
            totalReward += collisionPenalty;

            // Clamp and add the total reward
            AddReward(Mathf.Clamp(totalReward, -1f, 1f));

            // Update previous distance
            previousDistanceToTarget = distanceToTarget;
        }

        private void OnTriggerEnter(Collider other)
        {
            if (other.CompareTag("Entrance") && !reachedEntrance)
            {
                reachedEntrance = true; // Update flag to start focusing on parking spot
                AddReward(1.0f); // Large reward for reaching the entrance
                Debug.Log("Entrance reached!");
            }
            else if (other.CompareTag("ParkingSpot"))
            {
                AddReward(2.5f); // Large reward for successfully parking
                Debug.Log("Successfully Parked!");
                EndEpisode();
            }
        }

        // Apply throttle to the rear wheels
        private void ApplyThrottle(float throttle)
        {
            float motorTorque = throttle * 1500f; // Adjust for desired speed
            rearLeftWheel.motorTorque = motorTorque;
            rearRightWheel.motorTorque = motorTorque;
        }

        // Apply steering to the front wheels
        private void ApplySteering(float steering)
        {
            float steerAngle = steering * 25f;
            frontLeftWheel.steerAngle = steerAngle;
            frontRightWheel.steerAngle = steerAngle;
        }

        // Reset wheel colliders to their default states
        private void ResetWheelColliders()
        {
            frontLeftWheel.steerAngle = 0f;
            frontRightWheel.steerAngle = 0f;
            rearLeftWheel.motorTorque = 0f;
            rearRightWheel.motorTorque = 0f;
        }
        private void OnCollisionEnter(Collision collision)
        {
            if (collision.gameObject.CompareTag("Wall"))
            {
                collisionPenalty = -0.5f; // Large penalty for hitting the wall
                //Debug.Log("Collided with Wall!");
            }
            else if (collision.gameObject.CompareTag("Car"))
            {
                collisionPenalty = -0.5f; // Large penalty for hitting another car
                //Debug.Log("Collided with Car!");
            }
            else
            {
                collisionPenalty = 0f; // No penalty for other collisions
            }
        }


        private void OnCollisionStay(Collision collision)
        {
            // Continuously apply penalties while in contact
            if (collision.gameObject.CompareTag("Wall"))
            {
                collisionPenalty = -0.1f; // Smaller, repeated penalty for staying in contact with the wall
            }
            else if (collision.gameObject.CompareTag("Car"))
            {
                collisionPenalty = -0.1f; // Smaller, repeated penalty for staying in contact with another car
            }
        }


        private void OnCollisionExit(Collision collision)
        {
            // Reset penalty when collision ends
            if (collision.gameObject.CompareTag("Wall") || collision.gameObject.CompareTag("Car"))
            {
                collisionPenalty = 0f;
            }
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var continuousActions = actionsOut.ContinuousActions;
            continuousActions[0] = Input.GetAxis("Vertical"); // W/S or Up/Down
            continuousActions[1] = Input.GetAxis("Horizontal"); // A/D or Left/Right
        }
    }
}


