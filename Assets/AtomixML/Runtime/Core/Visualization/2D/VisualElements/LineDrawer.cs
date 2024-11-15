using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

namespace Atom.MachineLearning.Core.Visualization.VisualElements
{

    [RequireComponent(typeof(UIDocument))]
    public class LineDrawer : MonoBehaviour
    {
        [SerializeField] private float _drawSpeedPixelsPerSecond = 100f;
        [SerializeField] private List<Vector2> _linePoints = new List<Vector2>();
        private List<Vector2> _currentLinePoints = new List<Vector2>();
        private VisualElement _lineContainer;

        private void Start()
        {
            CreateUI();
            StartDrawingLine();
        }

        private void CreateUI()
        {
            var doc = GetComponent<UIDocument>();
            var root = doc.rootVisualElement;

            _lineContainer = new VisualElement();

            _lineContainer.style.width = 500;
            _lineContainer.style.height = 500;
            _lineContainer.style.backgroundColor = new StyleColor(Color.gray);

            root.Add(_lineContainer);
            _lineContainer.generateVisualContent += UpdateLine;

        }

        private void UpdateLine(MeshGenerationContext ctx)
        {
            if (_currentLinePoints.Count < 2)
            {
                return; // Exit if there are not enough points to form a path.
            }

            var painter2D = ctx.painter2D;
            painter2D.lineWidth = 5f;
            painter2D.strokeColor = Color.white;

            painter2D.BeginPath();
            painter2D.MoveTo(_currentLinePoints[0]);
            for (int i = 1; i < _currentLinePoints.Count; i++)
            {
                painter2D.LineTo(_currentLinePoints[i]);
            }

            painter2D.Stroke();

        }

        [ContextMenu("StartDrawingLine")]
        private void StartDrawingLine()
        {
            _lineContainer.Clear();
            StartCoroutine(MoveThroughPoints(_linePoints, _drawSpeedPixelsPerSecond, _lineContainer));
        }

        private IEnumerator MoveThroughPoints(List<Vector2> points, float speed, VisualElement visualElement)
        {
            if (points == null || points.Count < 2)
            {
                yield break; // Exit if there are not enough points to form a path.
            }

            _currentLinePoints.Clear();
            _currentLinePoints.Add(points[0]);

            // Move through each point in turn.
            for (int i = 0; i < points.Count - 1; i++)
            {
                Vector2 startPoint = points[i];
                Vector2 endPoint = points[i + 1];
                float journeyLength = Vector2.Distance(startPoint, endPoint);
                float startTime = Time.time;

                // Continue until we reach the end point.
                while (Vector2.Distance(_currentLinePoints[_currentLinePoints.Count - 1], endPoint) > 0.01f)
                {
                    float distCovered = (Time.time - startTime) * speed;
                    float fracJourney = distCovered / journeyLength;
                    Vector2 newPoint = Vector2.Lerp(startPoint, endPoint, fracJourney);

                    // Draw line
                    _currentLinePoints.Add(newPoint);
                    visualElement.MarkDirtyRepaint();

                    // Wait for the next frame.
                    yield return null;
                }

                // Ensure the end point is exact
                _currentLinePoints.Add(endPoint);
                visualElement.MarkDirtyRepaint();
            }
        }
    }
}