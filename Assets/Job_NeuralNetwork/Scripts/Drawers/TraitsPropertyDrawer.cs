using Assets.Job_NeuralNetwork.Scripts.GeneticNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;

namespace Assets.Job_NeuralNetwork.Scripts
{
    //[CustomPropertyDrawer(typeof(Gene))]

    class TraitsPropertyDrawer : PropertyDrawer
    {
     /*   public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            label = EditorGUI.BeginProperty(position, label, property);
            var indent = EditorGUI.indentLevel;
            EditorGUI.indentLevel = 0;
            EditorGUI.LabelField(new Rect(position.x, position.y, 10, position.height), "*");
            EditorGUI.PrefixLabel(position, GUIUtility.GetControlID(FocusType.Passive), label);
            position.height *= 2;

            //var t = property.FindPropertyRelative("TraitName");
            var m = property.FindPropertyRelative("MutationVersion");
            var v = property.FindPropertyRelative("Value");
            var d = property.FindPropertyRelative("Dominance");


            Rect labelFRect = new Rect(position.x, position.y, position.width * 0.29f, position.height*0.5f);
            Rect labelTRect = new Rect(position.x + position.width * 0.5f, position.y, position.width * 0.09f, position.height);
            Rect labelSRect = new Rect(position.x + position.width * 0.8f, position.y, position.width * 0.09f, position.height);

            Rect minRect = new Rect(position.x + position.width * 0.3f, position.y, position.width * 0.19f, position.height);
            Rect mirroredRect = new Rect(position.x + position.width * 0.6f, position.y, position.width * 0.19f, position.height);
            Rect maxRect = new Rect(position.x + position.width * 0.9f, position.y, position.width * 0.09f, position.height);

            EditorGUI.LabelField(labelFRect, "Trait Name");
          
            //EditorGUI.PropertyField(minRect, n, GUIContent.none);
            //EditorGUI.PropertyField(mirroredRect, t, GUIContent.none);
            //EditorGUI.PropertyField(maxRect, s, GUIContent.none);

            EditorGUI.EndProperty();
        }*/
    }
}
