//using System;
//using System.Collections;
//using System.Collections.Generic;
//using UnityEngine;


//namespace AutonomousParking
//{

//    public class carControl : MonoBehaviour
//    {
//        [SerializeField] private WheelCollider FrontLeftcollider, FrontRightcollider, RearLeftcollider, RearRightcollider;
//        // Start is called before the first frame update

//        IEnumerator DelaySeconds()
//        {
//            yield return new WaitForSeconds(3f);
//            FrontLeftcollider.brakeTorque = -50f;
//            FrontRightcollider.brakeTorque = -50f;
//            Debug.Log("Brake applied");
//            // FrontLeftcollider.motorTorque = 30;
//            // FrontRightcollider.motorTorque = 30;  
//        }
//        void Start()
//        {
//            FrontLeftcollider.motorTorque = 100;
//            FrontRightcollider.motorTorque = 100;
//            // StartCoroutine(DelaySeconds());

//        }

//        public void PerformAction(List<int> action)
//        {
//            SetTorque(action[0]);
//            SetSteerAngle(action[1]);
//            StopCar(action[2]);
//        }

//        // Update is called once per frame
//        void Update()
//        {

//        }

//        void SetTorque(int torque)
//        {
//            RearLeftcollider.motorTorque = torque;
//            RearRightcollider.motorTorque = torque;
//        }

//        void SetSteerAngle(int angle)
//        {
//            FrontLeftcollider.steerAngle = angle;
//            FrontRightcollider.steerAngle = angle;
//        }

//        void StopCar(int flag)
//        {
//            if (flag == 1)
//            {
//                RearLeftcollider.motorTorque = -3000;
//                RearRightcollider.motorTorque = -3000;
//            }
//        }
//    }
//}
