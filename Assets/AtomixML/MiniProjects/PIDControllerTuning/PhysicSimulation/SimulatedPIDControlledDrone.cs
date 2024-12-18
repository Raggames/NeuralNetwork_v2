using Sirenix.OdinInspector;
using System;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.PIDControllerTuning
{
    public class SimulatedPIDControlledDrone : MonoBehaviour
    {
        private Rigidbody _rigidbody;

        [SerializeField] private Transform _target;
        [SerializeField] private float _forceMultiplier = 1;
        [SerializeField] private float _torqueMultiplier = 1;

        [Header("Translation")]
        [SerializeField] private float _trs_P;
        [SerializeField] private float _trs_I;
        [SerializeField] private float _trs_D;
        [SerializeField] private int _trs_DRange;
        [SerializeField] private float _trs_maxRecordTime;

        [Header("Rotation")]
        [SerializeField] private float _rot_P;
        [SerializeField] private float _rot_I;
        [SerializeField] private float _rot_D;
        [SerializeField] private int _rot_DRange;
        [SerializeField] private float _rot_maxRecordTime;
        [Header("Engine")]
        [SerializeField] private float _maxEngineForce = 100;
        [SerializeField] private float _rotationSensivity = .33f;
        [ShowInInspector, ReadOnly] private float _currentPidVectorMagnitude;
        [ShowInInspector, ReadOnly] private Vector4 _engineThrust;

        [Space]
        [SerializeField] private PIDFunction[] _translationAxisPIDFunctions;
        [SerializeField] private PIDFunction[] _rotationAxisPIDFunctions;

        [Space]
        [SerializeField] private Transform _top_right_motor;
        [SerializeField] private Transform _top_left_motor;
        [SerializeField] private Transform _bottom_right_motor;
        [SerializeField] private Transform _bottom_left_motor;

        [Space]
        [SerializeField] private float _debugDrawrayScaling = .05f;
        [SerializeField] private bool _enableSimpleForceMode = true;


        private void Awake()
        {
            _rigidbody = GetComponent<Rigidbody>();

            for (int i = 0; i < _translationAxisPIDFunctions.Length; i++)
                _translationAxisPIDFunctions[i].SetTime(Time.fixedDeltaTime, _trs_maxRecordTime);

            for (int i = 0; i < _rotationAxisPIDFunctions.Length; ++i)
                _rotationAxisPIDFunctions[i].SetTime(Time.fixedDeltaTime, _rot_maxRecordTime);
        }

        private void FixedUpdate()
        {
            for (int i = 0; i < _translationAxisPIDFunctions.Length; i++)
            {
                _translationAxisPIDFunctions[i].SetParameters(_trs_P, _trs_I, _trs_D, _trs_DRange);
            }

            for (int i = 0; i < _rotationAxisPIDFunctions.Length; ++i)
            {
                _rotationAxisPIDFunctions[i].SetParameters(_rot_P, _rot_I, _rot_D, _rot_DRange);
            }

            var positionError = transform.position - _target.position;

            var pidVector = ComputeOffsetPIDVector(positionError);

            Stabilize();

            Stabilize2(positionError, pidVector);
        }


        /// <summary>
        /// Calculate the PID vector from the error of position
        /// This PID vector will serve to other functions to orient the drone/throttle the engines 
        /// </summary>
        /// <param name="offset"></param>
        /// <returns></returns>
        private Vector3 ComputeOffsetPIDVector(Vector3 offset)
        {
            // x force is handled by a difference of rotation with top and bottom engines
            var x_force = (float)_translationAxisPIDFunctions[0].Compute(offset.x, 0);

            // y force is handled by a difference of rotation between opposite engines (not simulated now)
            var y_force = (float)_translationAxisPIDFunctions[1].Compute(offset.y, 0);

            // z force is handled by a difference of rotation with left and right engines
            var z_force = (float)_translationAxisPIDFunctions[2].Compute(offset.z, 0);

            var pidForce = new Vector3(x_force, y_force, z_force);
            Debug.DrawRay(transform.position, pidForce, Color.red);

            _currentPidVectorMagnitude = (pidForce * _forceMultiplier).magnitude;

            if (_enableSimpleForceMode)
                _rigidbody.AddForceAtPosition(pidForce * _forceMultiplier, transform.position - Vector3.down);

            return new Vector3(x_force, y_force, z_force);
        }

        /// <summary>
        /// Simple version of stabilization, doesn't take in account the motors or so
        /// </summary>
        private void Stabilize()
        {
            // transform offset to an orientation quaternion
            // the drone always want to aim at its target
            var offsetR = transform.rotation.eulerAngles - _target.transform.eulerAngles;

            var xr_force = (float)_rotationAxisPIDFunctions[0].Compute(WrapAngle(offsetR.x), 0);
            var yr_force = (float)_rotationAxisPIDFunctions[1].Compute(WrapAngle(offsetR.y), 0);
            var zr_force = (float)_rotationAxisPIDFunctions[2].Compute(WrapAngle(offsetR.z), 0);

            if (_enableSimpleForceMode)
                _rigidbody.AddTorque(new Vector3(xr_force, yr_force, zr_force) * _torqueMultiplier);
        }

        private void Stabilize2(Vector3 positionError, Vector3 pidVector)
        {
            // the dot product ensure the drone orientation is aligned with the destination vector
            var dot = Vector3.Dot(transform.up, -positionError.normalized);
            var baseThrottle = Mathf.Clamp(pidVector.magnitude * _forceMultiplier, 0, _maxEngineForce);
            // if the drone is flipped comparated to the orientation of the pid vector, 
            // all the power will go into the orientation
            // else the two components are taken in account
            if (dot < 0)
                baseThrottle = 0;

            // engine thrust to position repartition
            // w, x
            // y, z
            _engineThrust = new Vector4(baseThrottle, baseThrottle, baseThrottle, baseThrottle);

            // roulis / left to right on Z
            var z_projection = Vector3.ProjectOnPlane(pidVector, transform.forward);
            z_projection = Vector3.ProjectOnPlane(z_projection, transform.up);
            //Debug.DrawRay(transform.position, z_projection, Color.blue);

            var z_dot = Vector3.Dot(z_projection.normalized, transform.right);
            HandleLeftRightEngines(baseThrottle, z_dot * z_projection.magnitude, ref _engineThrust);

            // tangage / front to back
            var x_projection = Vector3.ProjectOnPlane(pidVector, transform.right);
            x_projection = Vector3.ProjectOnPlane(x_projection, transform.up);
            //Debug.DrawRay(transform.position, x_projection, Color.yellow);

            var x_dot = Vector3.Dot(x_projection.normalized, transform.forward);
            HandleFrontBackEngines(baseThrottle, x_dot * x_projection.magnitude, ref _engineThrust);

            // lacet / left right on Y
            var y_projection = Vector3.ProjectOnPlane(pidVector, transform.up);
            y_projection = Vector3.ProjectOnPlane(y_projection, transform.forward);
            var y_dot = Vector3.Dot(y_projection.normalized, transform.right);

            Debug.DrawRay(transform.position, y_projection, Color.green);
            HandleDiagonalEngines(Math.Sign(y_dot), y_projection.magnitude * x_projection.magnitude); // we turn only if going forward or backward


            if (!_enableSimpleForceMode)
            {
                _rigidbody.AddForceAtPosition(_top_left_motor.position, transform.up * _engineThrust.w);
                _rigidbody.AddForceAtPosition(_top_right_motor.position, transform.up * _engineThrust.x);

                _rigidbody.AddForceAtPosition(_bottom_left_motor.position, transform.up * _engineThrust.y);
                _rigidbody.AddForceAtPosition(_bottom_right_motor.position, transform.up * _engineThrust.z);
            }
        }


        private void HandleFrontBackEngines(double baseThrottle, double orientation, ref Vector4 engineThrust)
        {
            ComputeThrottleForce(1.0, baseThrottle, orientation, out var positive, out var negative);

             Debug.DrawRay(_top_left_motor.position, transform.up * (float)orientation* _debugDrawrayScaling , Color.yellow);
             Debug.DrawRay(_top_right_motor.position, transform.up * (float)orientation* _debugDrawrayScaling, Color.yellow);

             Debug.DrawRay(_bottom_left_motor.position, transform.up * -(float)orientation* _debugDrawrayScaling, Color.yellow);
             Debug.DrawRay(_bottom_right_motor.position, transform.up * -(float)orientation* _debugDrawrayScaling, Color.yellow);


            /*Debug.DrawRay(_top_left_motor.position, transform.up * (float)positive * _debugDrawrayScaling / 4, Color.yellow);
            Debug.DrawRay(_top_right_motor.position, transform.up * (float)positive * _debugDrawrayScaling / 4, Color.yellow);

            Debug.DrawRay(_bottom_left_motor.position, transform.up * (float)negative * _debugDrawrayScaling / 4, Color.yellow);
            Debug.DrawRay(_bottom_right_motor.position, transform.up * (float)negative * _debugDrawrayScaling / 4, Color.yellow);*/


            engineThrust = new Vector4(
                engineThrust.w + (float)-orientation / 4,
                engineThrust.x + (float)-orientation / 4,
                engineThrust.y + (float)orientation / 4,
                engineThrust.z + (float)orientation / 4);


        }

        private void HandleLeftRightEngines(double baseThrottle, double orientation, ref Vector4 engineThrust)
        {
            Debug.DrawRay(_top_left_motor.position, transform.up * (float)orientation * _debugDrawrayScaling, Color.blue);
            Debug.DrawRay(_bottom_left_motor.position, transform.up * (float)orientation * _debugDrawrayScaling, Color.blue);

            Debug.DrawRay(_top_right_motor.position, transform.up * -(float)orientation * _debugDrawrayScaling, Color.blue);
            Debug.DrawRay(_bottom_right_motor.position, transform.up * -(float)orientation * _debugDrawrayScaling, Color.blue);

            /*Debug.DrawRay(_top_left_motor.position, transform.up * (float)positive / 4 * _debugDrawrayScaling, Color.blue);
            Debug.DrawRay(_bottom_left_motor.position, transform.up * (float)positive / 4 * _debugDrawrayScaling, Color.blue);

            Debug.DrawRay(_top_right_motor.position, transform.up * (float)negative / 4 * _debugDrawrayScaling, Color.blue);
            Debug.DrawRay(_bottom_right_motor.position, transform.up * (float)negative / 4 * _debugDrawrayScaling, Color.blue);*/

            engineThrust = new Vector4(
                engineThrust.w + (float)orientation * _rotationSensivity / 4,
                engineThrust.x + (float)-orientation * _rotationSensivity / 4,
                engineThrust.y + (float)orientation * _rotationSensivity / 4,
                engineThrust.z + (float)-orientation * _rotationSensivity / 4);
        }

        /// <summary>
        /// Rotating on Y axis is done by changing the speed of opposite engines
        /// The speed change must be symetric on diagonal (and inverted)
        /// </summary>
        /// <param name="direction"></param>
        /// <param name="throttle"></param>
        private void HandleDiagonalEngines(int direction, double throttle)
        {
            if (direction < 0)
            {
                Debug.DrawRay(_top_left_motor.position, transform.right * (float)throttle, Color.blue);
                Debug.DrawRay(_bottom_right_motor.position, transform.right * -(float)throttle, Color.blue);
            }
            else if (direction > 0)
            {
                Debug.DrawRay(_top_right_motor.position, transform.right * -(float)throttle, Color.blue);
                Debug.DrawRay(_bottom_left_motor.position, transform.right * (float)throttle, Color.blue);
            }
        }


        /// <summary>
        /// Throttle output must handle a base component (the need for poser to translate) and an orientation component 
        /// (the need to variate the throttle to orient the drone)
        /// We have to compute both so we use a function such as
        /// if throttle base is 80% and orient is 20%, one engine will output 100 and the opposite 60%
        /// if throttle base is 90% and orient is 20%, one engine will output 100 and the opposite 50% (the overshoot from a side is inverted and sent to the other)
        /// </summary>
        /// <param name="throttleBaseComponent"></param>
        /// <param name="throttleOrientationComponent"></param>
        /// <param name="result"></param>
        /// <param name="overshoot"></param>
        private void ComputeThrottleForce(double repartitionRatio, double throttleBaseComponent, double throttleOrientationComponent, out double positive, out double negative)
        {
            var throttleRatio = throttleBaseComponent * 100 / _maxEngineForce;
            var orientationRatio = throttleOrientationComponent * 100 / _maxEngineForce;
            Debug.Log(throttleRatio + " " + orientationRatio);

            var sum = throttleRatio + orientationRatio;
            var diff = 100.0 - repartitionRatio * sum;
            if (diff < 0)
            {
                negative = 100 - diff;
                positive = 100;
            }
            else
            {
                negative = 100 - sum;
                positive = 100 + sum;
            }
        }

        /*
         la prochaine étape est de fixer ma pousser à l'axe des moteurs et d'utiliser l'erreur d'orientation entre le vecteur de poussée que je veux 
        (le vecteur drone-cible rouge passé dans la fonction PID sur ces 3 axes) et l'orientation de la machine, 
        pour balancer les bonnes commandes aux "moteurs" pour orienter le drone dans l'axe de son vecteur de poussée
         */

        private float WrapAngle(float angle)
        {
            return angle > 180 ? angle - 360 : angle;
        }
    }

}
