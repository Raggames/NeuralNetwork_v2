using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Optimizers;
using Sirenix.OdinInspector;
using System;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.PIDControllerTuning
{
    public enum StabilizationModes
    {
        Proto,
        FollowAndOrient,
        Angle,
        AngularSpeed,
    }

    public class SimulatedPIDControlledDrone : MonoBehaviour, IGeneticEntity
    {
        private Rigidbody _rigidbody;

        [SerializeField] private Transform _target;

        [Space]
        [SerializeField] private bool _disableOnCollide = false;
        [SerializeField] private float _debugDrawrayScaling = .05f;
        [SerializeField] private StabilizationModes _stabilizationMode = StabilizationModes.Proto;

        [Header("Translation")]
        [SerializeField] private float _trs_P = 1.25f;
        [SerializeField] private float _trs_I = .45f;
        [SerializeField] private float _trs_D = .125f;
        [SerializeField] private int _trs_DRange;
        [SerializeField] private float _trs_maxRecordTime;

        [Header("Rotation")]
        [SerializeField] private float _rot_P = 1.25f;
        [SerializeField] private float _rot_I = .45f;
        [SerializeField] private float _rot_D = .125f;
        [SerializeField] private int _rot_DRange;
        [SerializeField] private float _rot_maxRecordTime;

        [Header("Engine")]
        [SerializeField] private bool allowNegativeEngineForces = true;
        [SerializeField] private float _thrustSmoothness = .05f;
        [SerializeField] private float _maxEngineForce = 100;
        [SerializeField] private float _offsetCorrectionRatio = 100;
        [SerializeField] private float _translationSensivity = .1f;
        [SerializeField] private float _rotationSensivity = .33f;
        [SerializeField] private float _velocityCompensationRatio = 1f;

        [ShowInInspector, ReadOnly] private float _currentPidVectorMagnitude;
        [ShowInInspector, ReadOnly] private Vector4 _engineThrust;

        [Header("Fit function")]
        /// Angle error threshold (computed by dotproduct) between orientation of drone and target 
        /// the more the agent is under threshold, the more points it gains
        [SerializeField] private float _rewardAngleThreshold = 5;
        /// <summary>
        /// Rewarding angle from target orientation under threshold over time
        /// </summary>
        [SerializeField] private float _rewardAngleValuePerSecond = 1;
        /// <summary>
        /// Penalizing distance from target over time
        /// </summary>
        [SerializeField] private float _penaltyPerDistanceFromTargetPerSecond = .1f;

        /// <summary>
        /// L'utilisation des moteurs donne un penalty à l'agent.
        /// Il faut qu'il trouve la fonction qui utilise le moins d'energie pour stabiliser son état
        /// </summary>
        [SerializeField] private float _powerPenaltyMultiplier = 5;

        [SerializeField] private float _engineThrustSignChangePenalty = .1f;

        [ShowInInspector, ReadOnly] private float _orientationErrorDotProduct = 0.0f;
        [ShowInInspector, ReadOnly] private float _currentReward;
        public float currentReward => _currentReward;

        [Header("PID Functions")]
        [SerializeField] private PIDFunction[] _translationAxisPIDFunctions;
        [SerializeField] private PIDFunction[] _rotationAxisPIDFunctions;

        [Header("Architecture")]
        [SerializeField] private Rigidbody _top_right_motor;
        [SerializeField] private Rigidbody _top_left_motor;
        [SerializeField] private Rigidbody _bottom_right_motor;
        [SerializeField] private Rigidbody _bottom_left_motor;


        private Vector4 _thrustTemp;
        private Vector4 _thrustVel;

        public float MaxEngineForce { get => _maxEngineForce; set => _maxEngineForce = value; }
        public int Generation { get; set; }

        [SerializeField] private NVector _genes;
        public NVector Genes
        {
            get => _genes; set
            {
                _genes = value;

                _trs_P = (float)_genes[0];
                _trs_I = (float)_genes[1];
                _trs_D = (float)_genes[2];
                _translationSensivity = (float)_genes[3];
                _rotationSensivity = (float)_genes[4];
                _trs_DRange = (int)_genes[5];
            }
        }

        private void Awake()
        {

            _rigidbody = GetComponent<Rigidbody>();

            for (int i = 0; i < _translationAxisPIDFunctions.Length; i++)
                _translationAxisPIDFunctions[i].SetTime(Time.fixedDeltaTime, _trs_maxRecordTime);

            for (int i = 0; i < _rotationAxisPIDFunctions.Length; ++i)
                _rotationAxisPIDFunctions[i].SetTime(Time.fixedDeltaTime, _rot_maxRecordTime);
        }

        private void OnCollisionEnter(Collision collision)
        {
            // bounds layer
            if (_disableOnCollide &&  collision.gameObject.layer == 10)
            {
                this.enabled = false;
            }
        }

        private void FixedUpdate()
        {
            UpdatePIDParameters();

            var offset = (transform.position - _target.position);

            switch (_stabilizationMode)
            {
                case StabilizationModes.Proto:
                    FollowTargetMode1(offset);
                    StabilizeMode1();
                    break;
                case StabilizationModes.FollowAndOrient:
                    UpdateFollowAndOrient(offset);
                    break;
                case StabilizationModes.Angle:
                    UpdateAngleMode();
                    break;
                case StabilizationModes.AngularSpeed:
                    break;
            }

            ComputeAgentReward(offset);
        }


        private void UpdatePIDParameters()
        {
            for (int i = 0; i < _translationAxisPIDFunctions.Length; i++)
            {
                _translationAxisPIDFunctions[i].SetParameters(_trs_P, _trs_I, _trs_D, _trs_DRange);
                _translationAxisPIDFunctions[i].SetTime(Time.fixedDeltaTime, _trs_maxRecordTime);
            }

            for (int i = 0; i < _rotationAxisPIDFunctions.Length; ++i)
            {
                _rotationAxisPIDFunctions[i].SetParameters(_rot_P, _rot_I, _rot_D, _rot_DRange);
                _rotationAxisPIDFunctions[i].SetTime(Time.fixedDeltaTime, _trs_maxRecordTime);
            }
        }

        #region Mode 1

        /// <summary>
        /// Calculate the PID vector from the error of position
        /// This PID vector will serve to other functions to orient the drone/throttle the engines 
        /// </summary>
        /// <param name="offset"></param>
        /// <returns></returns>
        private Vector3 FollowTargetMode1(Vector3 offset)
        {
            // x force is handled by a difference of rotation with top and bottom engines
            var x_force = (float)_translationAxisPIDFunctions[0].Compute(offset.x, 0);

            // y force is handled by a difference of rotation between opposite engines (not simulated now)
            var y_force = (float)_translationAxisPIDFunctions[1].Compute(offset.y, 0);

            // z force is handled by a difference of rotation with left and right engines
            var z_force = (float)_translationAxisPIDFunctions[2].Compute(offset.z, 0);

            var pidForce = new Vector3(x_force, y_force, z_force) * MaxEngineForce;
            Debug.DrawRay(transform.position, pidForce, Color.red);

            _rigidbody.AddForceAtPosition(pidForce, transform.position - Vector3.down);

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

            _rigidbody.AddTorque(new Vector3(xr_force, yr_force, zr_force) * _rotationSensivity);
        }

        #endregion

        #region Mode 2

        private void UpdateFollowAndOrient(Vector3 offset)
        {
            // engine thrust to position repartition
            // w, x
            // y, z
            _engineThrust = new Vector4(0, 0, 0, 0);

            // on clamp l'erreur pid pour éviter trop de range sur l'erreur
            //var positionError = offset.normalized * Math.Min(offset.magnitude, _maxEngineForce);
            var positionError = offset * _offsetCorrectionRatio;

            var trs_pidVector = ComputeTranslationCompensationPIDVectorMode2(positionError);
            trs_pidVector = trs_pidVector.normalized * Math.Min(trs_pidVector.magnitude, MaxEngineForce);

            FollowTargetMode2(positionError, trs_pidVector);

            /*var rot_pidVector = ComputeOrientationStabilizationPIDVectorMode2(positionError);
            rot_pidVector = rot_pidVector.normalized * Math.Min(rot_pidVector.magnitude, _maxEngineForce);
            StabilizeMode2(positionError, rot_pidVector);*/

            /*var rot_pidVector = ComputeOrientationStabilizationPIDVectorMode2(positionError);
            rot_pidVector = rot_pidVector.normalized * Math.Min(rot_pidVector.magnitude, MaxEngineForce);
*/
            StabilizeMode3(trs_pidVector);

            // todo clamping negative values on engines ? blade can turn in the opposite direction as well
            _thrustTemp.x = Mathf.SmoothDamp(_thrustTemp.x, _engineThrust.x, ref _thrustVel.x, _thrustSmoothness);
            _thrustTemp.y = Mathf.SmoothDamp(_thrustTemp.y, _engineThrust.y, ref _thrustVel.y, _thrustSmoothness);
            _thrustTemp.z = Mathf.SmoothDamp(_thrustTemp.z, _engineThrust.z, ref _thrustVel.z, _thrustSmoothness);
            _thrustTemp.w = Mathf.SmoothDamp(_thrustTemp.w, _engineThrust.w, ref _thrustVel.w, _thrustSmoothness);

            // apply forces to engines
            _rigidbody.AddForceAtPosition(transform.up * _thrustTemp.w, _top_left_motor.position);
            _rigidbody.AddForceAtPosition(transform.up * _thrustTemp.x, _top_right_motor.position);

            _rigidbody.AddForceAtPosition(transform.up * _thrustTemp.y, _bottom_left_motor.position);
            _rigidbody.AddForceAtPosition(transform.up * _thrustTemp.z, _bottom_right_motor.position);

            /*_top_left_motor.AddForce(transform.up * _engineThrust.w);
            _top_right_motor.AddForce(transform.up * _engineThrust.x);

            _bottom_left_motor.AddForce(transform.up * _engineThrust.y);
            _bottom_right_motor.AddForce(transform.up * _engineThrust.z);*/
        }

        private Vector3 ComputeOrientationStabilizationPIDVectorMode2(Vector3 offset)
        {
            // x force is handled by a difference of rotation with top and bottom engines
            var x_force = (float)_rotationAxisPIDFunctions[0].Compute(offset.x, 0);

            // y force is handled by a difference of rotation between opposite engines (not simulated now)
            var y_force = (float)_rotationAxisPIDFunctions[1].Compute(offset.y, 0);

            // z force is handled by a difference of rotation with left and right engines
            var z_force = (float)_rotationAxisPIDFunctions[2].Compute(offset.z, 0);

            //var pidForce = new Vector3(Math.Clamp(x_force, -_maxEngineForce, _maxEngineForce), Math.Clamp(y_force, -_maxEngineForce, _maxEngineForce), Math.Clamp(z_force, -_maxEngineForce, _maxEngineForce));
            var pidForce = new Vector3(x_force, y_force, z_force);

            pidForce -= _rigidbody.linearVelocity * _velocityCompensationRatio;

            Debug.DrawRay(transform.position, pidForce, Color.black);

            _currentPidVectorMagnitude = pidForce.magnitude;
            return new Vector3(x_force, y_force, z_force);
        }

        private Vector3 ComputeTranslationCompensationPIDVectorMode2(Vector3 offset)
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

        private void FollowTargetMode2(Vector3 positionError, Vector3 pidVector)
        {
            //pidVector = pidVector.normalized * Math.Min(pidVector.magnitude, _maxEngineForce);

            var baseThrottle = Vector4.zero;
            var dot = Vector3.Dot(transform.up, -positionError.normalized);
            if (dot > 0)
                baseThrottle = Vector4.one * pidVector.magnitude * _translationSensivity;
            // baseThrottle = Mathf.Sign(dot) * Vector4.one * pidVector.magnitude * _translationSensivity;

            _engineThrust += baseThrottle;
        }

        private void StabilizeMode2(Vector3 positionError, Vector3 pidVector)
        {
            //pidVector = pidVector.normalized * Math.Min(pidVector.magnitude, _maxEngineForce);

            var force_inverter = 1; // Math.Sign(Vector3.Dot(transform.up, -positionError.normalized)) > 0 ? 1 : 0;

            // roulis / left to right on Z
            var z_projection = Vector3.ProjectOnPlane(pidVector, transform.forward);
            z_projection = Vector3.ProjectOnPlane(z_projection, transform.up);
            var z_dot = Math.Sign(Vector3.Dot(z_projection.normalized, transform.right));
            // correcting the projections magnitude with maxEngineForce
            // z_projection.magnitude/_maxEngineForce becomes a ratio 0 > 1 of the throttle applied to each engine
            HandleLeftRightEngines(z_dot * z_projection.magnitude * force_inverter, ref _engineThrust);
            Debug.DrawRay(transform.position, z_projection, Color.blue);


            // tangage / front to back
            var x_projection = Vector3.ProjectOnPlane(pidVector, transform.right);
            x_projection = Vector3.ProjectOnPlane(x_projection, transform.up);
            var x_dot = Math.Sign(Vector3.Dot(x_projection.normalized, transform.forward));
            HandleFrontBackEngines(x_dot * x_projection.magnitude * force_inverter, ref _engineThrust);
            Debug.DrawRay(transform.position, x_projection, Color.yellow);

            // lacet / left right on Y
            //var y_projection = Vector3.ProjectOnPlane(pidVector, transform.up);
            var y_projection = Vector3.ProjectOnPlane(_target.forward, transform.up);
            y_projection = Vector3.ProjectOnPlane(y_projection, transform.forward);
            var y_dot = Math.Sign(Vector3.Dot(y_projection.normalized, transform.right));
            Debug.DrawRay(transform.position, y_projection, Color.green);

            //HandleDiagonalEngines(Math.Sign(y_dot), y_projection.magnitude * x_projection.magnitude, ref _engineThrust); // we turn only if going forward or backward
            HandleDiagonalEngines(y_dot, y_projection.magnitude, ref _engineThrust);
        }

        private void StabilizeMode3(Vector3 pidVector)
        {
            //pidVector = pidVector.normalized * Math.Min(pidVector.magnitude, _maxEngineForce);

            var force_inverter = 1; // Math.Sign(Vector3.Dot(transform.up, -positionError.normalized)) > 0 ? 1 : 0;

            // roulis / left to right on Z
            var z_projection = Vector3.ProjectOnPlane(pidVector, transform.forward);
            var z_angle_error = Vector3.SignedAngle(z_projection, transform.up, transform.forward);
            // correcting the projections magnitude with maxEngineForce
            // z_projection.magnitude/_maxEngineForce becomes a ratio 0 > 1 of the throttle applied to each engine
            Debug.DrawRay(transform.position, z_projection, Color.blue);

            // tangage / front to back
            var x_projection = Vector3.ProjectOnPlane(-pidVector, transform.right);
            var x_angle_error = Vector3.SignedAngle(x_projection, transform.up, transform.right);
            Debug.DrawRay(transform.position, x_projection, Color.yellow);

            // lacet / left right on Y
            var y_projection = Vector3.ProjectOnPlane(_target.forward, transform.up);
            var y_angle_error = Vector3.SignedAngle(y_projection, transform.forward, transform.up);

            Debug.DrawRay(transform.position, y_projection, Color.green);

            /*var command = ComputeOrientationStabilizationPIDVectorMode2(new Vector3(
                (float)MLMath.Map(-x_angle_error, -180, 180, -1, 1),
                (float)MLMath.Map(-y_angle_error, -180, 180, -1, 1),
                (float)MLMath.Map(-z_angle_error, -180, 180, -1, 1)));

            HandleLeftRightEngines(command.z, ref _engineThrust);
            HandleFrontBackEngines(command.x, ref _engineThrust);
            HandleDiagonalEngines(Math.Sign(command.y), Math.Abs(command.y), ref _engineThrust);*/

            HandleLeftRightEngines(z_angle_error, ref _engineThrust);
            HandleFrontBackEngines(x_angle_error, ref _engineThrust);

            HandleDiagonalEngines(Math.Sign(y_angle_error), Math.Abs(y_angle_error), ref _engineThrust);

            Debug.Log($"{x_angle_error}, {y_angle_error}, {z_angle_error}");
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

            /* if(orientation > 0)
             {
                 engineThrust = new Vector4(
                engineThrust.x + (float)orientation * _rotationSensivity / 4,
                0,
                0,
                engineThrust.w + (float)orientation * _rotationSensivity / 4);
             }
             else
             {
                 engineThrust = new Vector4(
                0,
                engineThrust.y + (float)orientation * _rotationSensivity / 4,
                engineThrust.z + (float)orientation * _rotationSensivity / 4,
                0);
             }*/

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

        #endregion

        #region Mode Angle

        [ShowInInspector, ReadOnly] private Vector3 _targetEulerAngles;

        /// <summary>
        /// Apply a target angle on axis
        /// </summary>
        /// <param name="eulerAngles"></param>
        public void SetAnglesTarget(Vector3 eulerAngles)
        {

        }

        private void UpdateAngleMode()
        {
            //Debug.DrawRay(transform.position, _targetEulerAngles, Color.blue);
            _engineThrust = Vector4.zero;

            // right to left
            var z_angle_error = Vector3.SignedAngle(_target.right, transform.right, transform.forward);
            // tangage / front to back
            var x_angle_error = Vector3.SignedAngle(_target.forward, -transform.forward, transform.right);
            // lacet / left right on Y
            var y_angle_error = Vector3.SignedAngle(_target.right, transform.right, transform.up);

            // x force is handled by a difference of rotation with top and bottom engines
            var x_force = (float)_rotationAxisPIDFunctions[0].Compute(-x_angle_error, 0);

            // y force is handled by a difference of rotation between opposite engines (not simulated now)
            var y_force = (float)_rotationAxisPIDFunctions[1].Compute(-y_angle_error, 0);

            // z force is handled by a difference of rotation with left and right engines
            var z_force = (float)_rotationAxisPIDFunctions[2].Compute(-z_angle_error, 0);

            var pidForce = new Vector3(x_force, y_force, z_force);
            pidForce = pidForce.normalized * Math.Min(pidForce.magnitude, MaxEngineForce);

            Debug.DrawRay(transform.position, pidForce, Color.red);

            //HandleDiagonalEngines(Math.Sign(pidForce.y), Math.Abs(pidForce.y), ref _engineThrust);
            HandleFrontBackEngines(x_force, ref _engineThrust);
            //HandleLeftRightEngines(z_force, ref _engineThrust);

            _thrustTemp.x = Mathf.SmoothDamp(_thrustTemp.x, _engineThrust.x, ref _thrustVel.x, _thrustSmoothness);
            _thrustTemp.y = Mathf.SmoothDamp(_thrustTemp.y, _engineThrust.y, ref _thrustVel.y, _thrustSmoothness);
            _thrustTemp.z = Mathf.SmoothDamp(_thrustTemp.z, _engineThrust.z, ref _thrustVel.z, _thrustSmoothness);
            _thrustTemp.w = Mathf.SmoothDamp(_thrustTemp.w, _engineThrust.w, ref _thrustVel.w, _thrustSmoothness);

            // apply forces to engines
            _rigidbody.AddForceAtPosition(transform.up * _thrustTemp.w, _top_left_motor.position);
            _rigidbody.AddForceAtPosition(transform.up * _thrustTemp.x, _top_right_motor.position);

            _rigidbody.AddForceAtPosition(transform.up * _thrustTemp.y, _bottom_left_motor.position);
            _rigidbody.AddForceAtPosition(transform.up * _thrustTemp.z , _bottom_right_motor.position);

        }


        #endregion


        #region Training Heuristic & Management

        private void ComputeAgentReward(Vector3 offset)
        {
            // angle with target
            // we give reward when the agent orientation is close to the aiming target
            _orientationErrorDotProduct = Vector3.Dot(transform.up, -offset.normalized);
            var angle = 1f - _orientationErrorDotProduct;
            if (angle < _rewardAngleThreshold)
                _currentReward += _rewardAngleValuePerSecond * Time.fixedDeltaTime;

            var distance_error = offset.magnitude * _penaltyPerDistanceFromTargetPerSecond * Time.fixedDeltaTime;
            _currentReward -= distance_error;

            // penalty of power usage
            // as we search for a stable function, the use of engines (represented by the engine thrust) is used to penalize the agent
            // the goal is to find a way to achieve minimal error with minimal energy (high energy can achieve that first objective  but with a LOT of 'vibrations', if Integral and Derivative have high values)
            _currentReward -= _engineThrust.magnitude * _powerPenaltyMultiplier * Time.fixedDeltaTime;

            // penalizing with values of PID  ? is it a good idea ?

            // penalize with thrust axis changing signs (to count the frequency of overshoot//undershoot) ? 
            InspectThrustVector();
        }

        private Vector4 _previousThrustVector;

        private void InspectThrustVector()
        {
            if (Math.Sign(_previousThrustVector.x) != Math.Sign(_engineThrust.x))
                _currentReward -= _engineThrustSignChangePenalty;

            if (Math.Sign(_previousThrustVector.y) != Math.Sign(_engineThrust.y))
                _currentReward -= _engineThrustSignChangePenalty;

            if (Math.Sign(_previousThrustVector.z) != Math.Sign(_engineThrust.z))
                _currentReward -= _engineThrustSignChangePenalty;

            if (Math.Sign(_previousThrustVector.w) != Math.Sign(_engineThrust.w))
                _currentReward -= _engineThrustSignChangePenalty;

            _previousThrustVector = _engineThrust;
        }

        public double MutateGene(int geneIndex)
        {
            if (geneIndex == 5)
            {
                return MLRandom.Shared.Range(2, 50);
            }
            else
            {
                return _genes[geneIndex] + MLRandom.Shared.Range(-.025f, .025f);
            }
        }

        #endregion

        private float WrapAngle(float angle)
        {
            return angle > 180 ? angle - 360 : angle;
        }

    }

}