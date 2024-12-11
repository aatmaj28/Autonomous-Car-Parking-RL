using System.Collections.Generic;
using UnityEngine;

namespace AutonomousParking
{
    public class Spawner : MonoBehaviour
    {
        [SerializeField] private List<GameObject> prefabsToSpawn;
        private List<GameObject> spawnedCars = new List<GameObject>();

        private Vector3 firstRowStart = new Vector3(-6.98f, 0, 1.58f); // Starting position for the first row
        private Vector3 secondRowStart = new Vector3(4.23f, 0, 1.58f); // Starting position for the second row
        private float spotSpacing = 2.47f; // Space between parking spots
        private int spotsPerRow = 6; // Number of spots per row

        private int emptySpotIndex; // Index of the random empty spot
        private List<Transform> parkingSpots = new List<Transform>(); // List of parking spots

        void Start()
        {
            Random.InitState(System.DateTime.Now.Millisecond);  // Ensures randomness based on the current time
            SpawnVehicles();
        }

        private void RemoveExistingPrefabs()
        {
            foreach (GameObject obj in spawnedCars)
            {
                Destroy(obj);
            }
            spawnedCars.Clear();
        }

        public void SpawnVehicles()
        {
            RemoveExistingPrefabs();

            // Randomize the empty spot index for this episode
            emptySpotIndex = Random.Range(0, spotsPerRow * 2); // 12 spots in total (2 rows of 6 spots)

            // Log the empty spot index for debugging
            Debug.Log($"Empty spot index for this episode: {emptySpotIndex}");

            // Clear and repopulate the parking spots list
            parkingSpots.Clear();

            // Loop through both rows (2 rows, 6 spots each)
            for (int row = 0; row < 2; row++)
            {
                Vector3 spawnPosition = row == 0 ? firstRowStart : secondRowStart;
                Quaternion spawnRotation = row == 0 ? Quaternion.Euler(0, -90, 0) : Quaternion.Euler(0, 90, 0);

                for (int spot = 0; spot < spotsPerRow; spot++)
                {
                    int currentSpotIndex = row * spotsPerRow + spot;

                    // Skip the randomized empty spot
                    if (currentSpotIndex == emptySpotIndex)
                    {
                        spawnPosition.z -= spotSpacing;
                        continue;
                    }

                    // Choose a random prefab
                    GameObject prefabToSpawn = GetRandomChoice(prefabsToSpawn);

                    // Instantiate the car prefab at the calculated position and rotation
                    GameObject spawnedCar = Instantiate(prefabToSpawn, spawnPosition, spawnRotation);
                    spawnedCars.Add(spawnedCar);

                    // Add BoxCollider to the spawned car if it doesn't already have one
                    AddBoxCollider(spawnedCar);

                    // Move the spawn position for the next spot
                    spawnPosition.z -= spotSpacing;

                    // Save parking spots (assuming you are using a Transform for parking spots)
                    Transform parkingSpot = new GameObject($"ParkingSpot_{currentSpotIndex}").transform;
                    parkingSpot.position = spawnPosition;
                    parkingSpots.Add(parkingSpot); // Make sure to add the spot to the list after it's created
                }
            }

            // Double-check that the parkingSpots list has the correct number of spots
            Debug.Log($"Total parking spots: {parkingSpots.Count}");
        }

        private static T GetRandomChoice<T>(List<T> items)
        {
            int index = Random.Range(0, items.Count);
            return items[index];
        }

        private void AddBoxCollider(GameObject car)
        {
            BoxCollider boxCollider = car.GetComponent<BoxCollider>();
            if (boxCollider == null)
            {
                boxCollider = car.AddComponent<BoxCollider>();
            }
            boxCollider.center = new Vector3(0, 0.6304449f, 0);
            boxCollider.size = new Vector3(1.576153f, 1.258026f, 3.916647f);
        }

        // New method to get the random parking spot
        public Transform GetRandomParkingSpot()
        {
            // Make sure that the parkingSpots list is populated correctly
            if (parkingSpots.Count > 0)
            {
                return parkingSpots[emptySpotIndex]; // Return the parking spot at the index of the empty spot
            }

            // Return null if no parking spots have been assigned yet
            Debug.LogError("Parking spots are empty or not populated correctly!");
            return null;
        }
    }
}
