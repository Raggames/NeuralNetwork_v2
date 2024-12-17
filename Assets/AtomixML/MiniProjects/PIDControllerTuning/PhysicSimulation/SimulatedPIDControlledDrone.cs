using Sirenix.OdinInspector;
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

        [Space]
        
        [SerializeField] private PIDFunction[] _translationAxisPIDFunctions;
        [SerializeField] private PIDFunction[] _rotationAxisPIDFunctions;

        [Space]
        [SerializeField] private Transform _top_right_motor;
        [SerializeField] private Transform _top_left_motor;
        [SerializeField] private Transform _bottom_right_motor;
        [SerializeField] private Transform _bottom_left_motor;



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

            var offset = transform.position - _target.position;

            // the dot product ensure the drone orientation is aligned with the destination vector
            var dot = Vector3.Dot(transform.up, -offset.normalized);

            FollowTarget(offset);

            Stabilize();
        }

        private void FollowTarget(Vector3 offset)
        {
            // x force is handled by a difference of rotation with top and bottom engines
            var x_force = (float)_translationAxisPIDFunctions[0].Compute(offset.x, 0);

            // y force is handled by a difference of rotation between opposite engines (not simulated now)
            var y_force = (float)_translationAxisPIDFunctions[1].Compute(offset.y, 0);

            // z force is handled by a difference of rotation with left and right engines
            var z_force = (float)_translationAxisPIDFunctions[2].Compute(offset.z, 0);

            var pidForce = new Vector3(x_force, y_force, z_force);
            Debug.DrawRay(transform.position, pidForce, Color.red);

            _rigidbody.AddForceAtPosition(pidForce * _forceMultiplier, transform.position - Vector3.down);

            var projection = Vector3.ProjectOnPlane(pidForce, transform.up);
            Debug.DrawRay(transform.position, projection, Color.blue);

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

            //_rigidbody.AddForceAtPosition(Vector3.right, new Vector3(0, zr_force, 0) * _torqueMultiplier);
            _rigidbody.AddTorque(new Vector3(xr_force, yr_force, zr_force) * _torqueMultiplier);

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
