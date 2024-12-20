using Atom.MachineLearning.Core.Maths;
using Sirenix.OdinInspector;
using System;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.PIDControllerTuning
{
    public class SimulatedPIDControlledDrone : MonoBehaviour
    {
        private Rigidbody _rigidbody;

        [SerializeField] private Transform _target;
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

        [Header("Fit function")]
        /// l'erreur d'angle limite pour accorder du reward à l'agent
        /// plus l'agent passe de temps en dessous de l'angle limite (entre son transform.up et le vecteur cible), plus il gagne de point
        [SerializeField] private float _rewardAngleThreshold = 5;
        /// <summary>
        /// Nombre de points par seconde gagnés en dessous du threshold
        /// </summary>
        [SerializeField] private float _rewardAngleValuePerSecond = 1;

        /// <summary>
        /// L'utilisation des moteurs donne un penalty à l'agent.
        /// Il faut qu'il trouve la fonction qui utilise le moins d'energie pour stabiliser son état
        /// </summary>
        [SerializeField] private float _powerPenaltyMultiplier = 5;

        [ShowInInspector, ReadOnly] private float _orientationErrorDotProduct = 0.0f;
        [ShowInInspector, ReadOnly] private float _currentReward;

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
        [SerializeField] private bool _enableMode1 = true;


        private void Awake()
        {
            _rigidbody = GetComponent<Rigidbody>();

            for (int i = 0; i < _translationAxisPIDFunctions.Length; i++)
                _translationAxisPIDFunctions[i].SetTime(Time.fixedDeltaTime, _trs_maxRecordTime);

            for (int i = 0; i < _rotationAxisPIDFunctions.Length; ++i)
                _rotationAxisPIDFunctions[i].SetTime(Time.fixedDeltaTime, _rot_maxRecordTime);
        }

        [SerializeField] private bool _animateTargetMove = false;
        [SerializeField] private float _targetMoveRange;
        [SerializeField] private float _targetSpeed;
        [SerializeField] private Vector3 _targetBasePosition = new Vector3(0, 3, 0);
        private Vector3 _targetCurrentPosition = Vector3.zero;
        private Vector3 _targetVelocity = Vector3.zero;

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

            var offset = (transform.position - _target.position);

            if (_enableMode1)
            {
                var pidVector = ComputeOffsetPIDVectorMode1(offset);

                StabilizeMode1();
            }
            else
            {
                // on clamp l'erreur pid pour éviter trop de range sur l'erreur
                var positionError = offset.normalized * Math.Min(offset.magnitude, _maxEngineForce);

                var pidVector = ComputeOffsetPIDVectorMode2(positionError);
                StabilizeMode2(positionError, pidVector);
            }

            if (_animateTargetMove)
            {
                var crt = (_target.position - _targetCurrentPosition).magnitude;
                if (crt < .01)
                {
                    _targetCurrentPosition = UnityEngine.Random.insideUnitSphere * _targetMoveRange + _targetBasePosition;
                }

                _target.position = Vector3.SmoothDamp(_target.position, _targetCurrentPosition, ref _targetVelocity, _targetSpeed);
            }

            ComputeAgentReward(offset);
        }

        private void ComputeAgentReward(Vector3 offset)
        {
            // angle with target
            // we give reward when the agent orientation is close to the aiming target
            _orientationErrorDotProduct = Vector3.Dot(transform.up, -offset.normalized);
            var angle = 1f - _orientationErrorDotProduct;
            if (angle < _rewardAngleThreshold)
                _currentReward += _rewardAngleValuePerSecond * Time.fixedDeltaTime;

            // penalty of power usage
            // as we search for a stable function, the use of engines (represented by the engine thrust) is used to penalize the agent
            // the goal is to find a way to achieve minimal error with minimal energy (high energy can achieve that first objective  but with a LOT of 'vibrations', if Integral and Derivative have high values)
            _currentReward -= _engineThrust.magnitude * _powerPenaltyMultiplier * Time.fixedDeltaTime;

            // penalizing with values of PID  ? is it a good idea ?

            // penalize with thrust axis changing signs (to count the frequency of overshoot//undershoot) ? 
        }

        /// <summary>
        /// Calculate the PID vector from the error of position
        /// This PID vector will serve to other functions to orient the drone/throttle the engines 
        /// </summary>
        /// <param name="offset"></param>
        /// <returns></returns>
        private Vector3 ComputeOffsetPIDVectorMode1(Vector3 offset)
        {
            // x force is handled by a difference of rotation with top and bottom engines
            var x_force = (float)_translationAxisPIDFunctions[0].Compute(offset.x, 0);

            // y force is handled by a difference of rotation between opposite engines (not simulated now)
            var y_force = (float)_translationAxisPIDFunctions[1].Compute(offset.y, 0);

            // z force is handled by a difference of rotation with left and right engines
            var z_force = (float)_translationAxisPIDFunctions[2].Compute(offset.z, 0);

            var pidForce = new Vector3(x_force, y_force, z_force);
            Debug.DrawRay(transform.position, pidForce, Color.red);

            _rigidbody.AddForceAtPosition(pidForce * _maxEngineForce, transform.position - Vector3.down);

            _currentPidVectorMagnitude = (pidForce * _maxEngineForce).magnitude;
            return new Vector3(x_force, y_force, z_force);
        }

        private Vector3 ComputeOffsetPIDVectorMode2(Vector3 offset)
        {
            // x force is handled by a difference of rotation with top and bottom engines
            var x_force = (float)_translationAxisPIDFunctions[0].Compute(offset.x, 0);

            // y force is handled by a difference of rotation between opposite engines (not simulated now)
            var y_force = (float)_translationAxisPIDFunctions[1].Compute(offset.y, 0);

            // z force is handled by a difference of rotation with left and right engines
            var z_force = (float)_translationAxisPIDFunctions[2].Compute(offset.z, 0);

            //var pidForce = new Vector3(Math.Clamp(x_force, -_maxEngineForce, _maxEngineForce), Math.Clamp(y_force, -_maxEngineForce, _maxEngineForce), Math.Clamp(z_force, -_maxEngineForce, _maxEngineForce));
            var pidForce = new Vector3(x_force, y_force, z_force);
            Debug.DrawRay(transform.position, pidForce, Color.red);

            _currentPidVectorMagnitude = pidForce.magnitude;
            return new Vector3(x_force, y_force, z_force);
        }

        /// <summary>
        /// Simple version of stabilization, doesn't take in account the motors or so
        /// </summary>
        private void StabilizeMode1()
        {
            // transform offset to an orientation quaternion
            // the drone always want to aim at its target
            var offsetR = transform.rotation.eulerAngles - _target.transform.eulerAngles;

            var xr_force = (float)_rotationAxisPIDFunctions[0].Compute(WrapAngle(offsetR.x), 0);
            var yr_force = (float)_rotationAxisPIDFunctions[1].Compute(WrapAngle(offsetR.y), 0);
            var zr_force = (float)_rotationAxisPIDFunctions[2].Compute(WrapAngle(offsetR.z), 0);

            if (_enableMode1)
                _rigidbody.AddTorque(new Vector3(xr_force, yr_force, zr_force) * _torqueMultiplier);
        }

        private void StabilizeMode2(Vector3 positionError, Vector3 pidVector)
        {
            /*// the dot product ensure the drone orientation is aligned with the destination vector
            var dot = Vector3.Dot(transform.up, -positionError.normalized);

            // if the drone is flipped comparated to the orientation of the pid vector, 
            // all the power will go into the orientation
            // else the two components are taken in account
            if (dot < 0)
                baseThrottle = 0;*/

            // engine thrust to position repartition
            // w, x
            // y, z
            _engineThrust = new Vector4(0, 0, 0, 0);

            // roulis / left to right on Z
            var z_projection = Vector3.ProjectOnPlane(pidVector, transform.forward);
            z_projection = Vector3.ProjectOnPlane(z_projection, transform.up);
            Debug.DrawRay(transform.position, z_projection, Color.blue);

            var z_dot = Math.Sign(Vector3.Dot(z_projection.normalized, transform.right));
            HandleLeftRightEngines(z_dot * z_projection.magnitude, ref _engineThrust);

            // tangage / front to back
            var x_projection = Vector3.ProjectOnPlane(pidVector, transform.right);
            x_projection = Vector3.ProjectOnPlane(x_projection, transform.up);
            Debug.DrawRay(transform.position, x_projection, Color.yellow);

            var x_dot = Math.Sign(Vector3.Dot(x_projection.normalized, transform.forward));
            HandleFrontBackEngines(x_dot * x_projection.magnitude, ref _engineThrust);

            // lacet / left right on Y
            //var y_projection = Vector3.ProjectOnPlane(pidVector, transform.up);
            var y_projection = Vector3.ProjectOnPlane(_target.forward, transform.up);
            y_projection = Vector3.ProjectOnPlane(y_projection, transform.forward);
            Debug.DrawRay(transform.position, y_projection, Color.green);

            var y_dot = Math.Sign(Vector3.Dot(y_projection.normalized, transform.right));
            //HandleDiagonalEngines(Math.Sign(y_dot), y_projection.magnitude * x_projection.magnitude, ref _engineThrust); // we turn only if going forward or backward
            HandleDiagonalEngines(y_dot, y_projection.magnitude, ref _engineThrust);



            /*                Debug.DrawRay(_top_left_motor.position + transform.forward * .33f, transform.up * _engineThrust.w * _debugDrawrayScaling, Color.black);
                            Debug.DrawRay(_top_right_motor.position + transform.forward * .33f, transform.up * _engineThrust.x * _debugDrawrayScaling, Color.black);

                            Debug.DrawRay(_bottom_left_motor.position + transform.forward * -.33f, transform.up * _engineThrust.y * _debugDrawrayScaling, Color.black);
                            Debug.DrawRay(_bottom_right_motor.position + transform.forward * -.33f, transform.up * _engineThrust.z * _debugDrawrayScaling, Color.black);

            */
            _rigidbody.AddForceAtPosition(transform.up * _engineThrust.w, _top_left_motor.position);
            _rigidbody.AddForceAtPosition(transform.up * _engineThrust.x, _top_right_motor.position);

            _rigidbody.AddForceAtPosition(transform.up * _engineThrust.y, _bottom_left_motor.position);
            _rigidbody.AddForceAtPosition(transform.up * _engineThrust.z, _bottom_right_motor.position);
        }


        private void HandleFrontBackEngines(double orientation, ref Vector4 engineThrust)
        {
            Debug.DrawRay(_top_left_motor.position, transform.up * -(float)orientation * _rotationSensivity * _debugDrawrayScaling, Color.yellow);
            Debug.DrawRay(_top_right_motor.position, transform.up * -(float)orientation * _rotationSensivity * _debugDrawrayScaling, Color.yellow);

            Debug.DrawRay(_bottom_left_motor.position, transform.up * (float)orientation * _rotationSensivity * _debugDrawrayScaling, Color.yellow);
            Debug.DrawRay(_bottom_right_motor.position, transform.up * (float)orientation * _rotationSensivity * _debugDrawrayScaling, Color.yellow);

            /*Debug.DrawRay(_top_left_motor.position, transform.up * (float)positive * _debugDrawrayScaling / 4, Color.yellow);
            Debug.DrawRay(_top_right_motor.position, transform.up * (float)positive * _debugDrawrayScaling / 4, Color.yellow);

            Debug.DrawRay(_bottom_left_motor.position, transform.up * (float)negative * _debugDrawrayScaling / 4, Color.yellow);
            Debug.DrawRay(_bottom_right_motor.position, transform.up * (float)negative * _debugDrawrayScaling / 4, Color.yellow);*/


            engineThrust = new Vector4(
                engineThrust.x + (float)-orientation * _rotationSensivity / 4,
                engineThrust.y + (float)orientation * _rotationSensivity / 4,
                engineThrust.z + (float)orientation * _rotationSensivity / 4,
                engineThrust.w + (float)-orientation * _rotationSensivity / 4);


        }

        private void HandleLeftRightEngines(double orientation, ref Vector4 engineThrust)
        {
            Debug.DrawRay(_top_left_motor.position, transform.up * (float)orientation * _rotationSensivity * _debugDrawrayScaling, Color.blue);
            Debug.DrawRay(_top_right_motor.position, transform.up * -(float)orientation * _rotationSensivity * _debugDrawrayScaling, Color.blue);

            Debug.DrawRay(_bottom_left_motor.position, transform.up * (float)orientation * _rotationSensivity * _debugDrawrayScaling, Color.blue);
            Debug.DrawRay(_bottom_right_motor.position, transform.up * -(float)orientation * _rotationSensivity * _debugDrawrayScaling, Color.blue);

            /*Debug.DrawRay(_top_left_motor.position, transform.up * (float)positive / 4 * _debugDrawrayScaling, Color.blue);
            Debug.DrawRay(_bottom_left_motor.position, transform.up * (float)positive / 4 * _debugDrawrayScaling, Color.blue);

            Debug.DrawRay(_top_right_motor.position, transform.up * (float)negative / 4 * _debugDrawrayScaling, Color.blue);
            Debug.DrawRay(_bottom_right_motor.position, transform.up * (float)negative / 4 * _debugDrawrayScaling, Color.blue);*/

            engineThrust = new Vector4(
                engineThrust.x + (float)-orientation * _rotationSensivity / 4,
                engineThrust.y + (float)orientation * _rotationSensivity / 4,
                engineThrust.z + (float)-orientation * _rotationSensivity / 4,
                engineThrust.w + (float)orientation * _rotationSensivity / 4);
        }

        /// <summary>
        /// Rotating on Y axis is done by changing the speed of opposite engines
        /// The speed change must be symetric on diagonal (and inverted)
        /// </summary>
        /// <param name="direction"></param>
        /// <param name="throttle"></param>
        private void HandleDiagonalEngines(int direction, double throttle, ref Vector4 engineThrust)
        {
            if (direction < 0)
            {
                Debug.DrawRay(_top_left_motor.position, transform.right * (float)throttle, Color.blue);
                Debug.DrawRay(_bottom_right_motor.position, transform.right * -(float)throttle, Color.blue);

                /*engineThrust = new Vector4(
                engineThrust.x,
                engineThrust.y,
                engineThrust.z + (float)-throttle * _rotationSensivity / 4,
                engineThrust.w + (float)throttle * _rotationSensivity / 4);*/

                //w
                _rigidbody.AddForceAtPosition(transform.right * (float)throttle * _rotationSensivity, _top_left_motor.position);
                // z
                _rigidbody.AddForceAtPosition(transform.right * (float)-throttle * _rotationSensivity, _bottom_right_motor.position);
            }
            else if (direction > 0)
            {
                Debug.DrawRay(_top_right_motor.position, transform.right * -(float)throttle, Color.blue);
                Debug.DrawRay(_bottom_left_motor.position, transform.right * (float)throttle, Color.blue);

                /* engineThrust = new Vector4(
                 engineThrust.x + (float)-throttle * _rotationSensivity / 4,
                 engineThrust.y + (float)throttle * _rotationSensivity / 4,
                 engineThrust.z,
                 engineThrust.w);*/

                // x
                _rigidbody.AddForceAtPosition(transform.right * (float)-throttle * _rotationSensivity, _top_right_motor.position);
                //y
                _rigidbody.AddForceAtPosition(transform.right * (float)throttle * _rotationSensivity, _bottom_left_motor.position);

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
