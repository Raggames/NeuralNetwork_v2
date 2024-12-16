using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Visualization.VisualElements;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UIElements;

namespace Atom.MachineLearning.Core.Visualization.VisualElements
{
    /// <summary>
    /// A simple graphic to input
    /// </summary>
    public class SimpleLineChart : AtomMLChart
    {
        private double[] _pointsY;
        private double[,] _pointsXY;
        private Func<List<double>> _getYValuesDelegates;
        private Func<List<Vector2>> _getXYValuesDelegates;

        private float _lineWidth;

        private Color _strokeColor = Color.black;
        public Color strokeColor { get { return _strokeColor; } set { _strokeColor = value; } }

        private Color _backgroundColor = Color.white;
        public Color backgroundColor { get { return _backgroundColor; } set { _backgroundColor = value; style.backgroundColor = new StyleColor(_backgroundColor); } }


        /// <summary>
        /// Unidimensional mode, the points will be placed by the maximum avalaible interval on X axis
        /// If 500 px and 500 points, 1 point per pixel on X
        /// </summary>
        /// <param name="getPoints"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        public SimpleLineChart(double[] pointsY, float lineWidth = 2f, int width = 300, int height = 300)
        {
            _pointsY = pointsY;
            _lineWidth = lineWidth;

            style.width = width;
            style.height = height;
            backgroundColor = _backgroundColor;

            generateVisualContent += GenerateLineY;
            generateVisualContent += DrawOrthonormalLines;
        }

        public SimpleLineChart(double[,] pointXY, float lineWidth = 2f, int width = 300, int height = 300)
        {
            _pointsXY = pointXY;
            _lineWidth = lineWidth;

            style.width = width;
            style.height = height;
            backgroundColor = _backgroundColor;

            generateVisualContent += GenerateLineXY;
            generateVisualContent += DrawOrthonormalLines;
        }

        public SimpleLineChart(Func<List<double>> getValuesDelegate, float lineWidth = 2f, int width = 300, int height = 300)
        {
            _getYValuesDelegates = getValuesDelegate;
            _lineWidth = lineWidth;

            style.width = width;
            style.height = height;
            backgroundColor = _backgroundColor;

            generateVisualContent += GenerateLineYDynamic;
            generateVisualContent += DrawOrthonormalLines;
        }

        public SimpleLineChart(Func<List<Vector2>> getValuesDelegate, float lineWidth = 2f, int width = 300, int height = 300)
        {
            _getXYValuesDelegates = getValuesDelegate;
            _lineWidth = lineWidth;

            style.width = width;
            style.height = height;
            backgroundColor = _backgroundColor;

            generateVisualContent += GenerateLineXYDynamic;
            generateVisualContent += DrawOrthonormalLines;
        }


        /// <summary>
        /// Generate the line without knowing any x value, so we assume a equal distribution of points on x and just compute the interval by pointsCount / avalaibleWidth 
        /// </summary>
        /// <param name="ctx"></param>
        protected void GenerateLineY(MeshGenerationContext ctx)
        {
            var painter2D = ctx.painter2D;

            painter2D.lineWidth = _lineWidth;
            painter2D.strokeColor = strokeColor;

            MLMath.ColumnMinMax(_pointsY, out y_min, out y_max);

            x_min = 0;
            x_max = _pointsY.Length;

            painter2D.BeginPath();

            var relative_position_x = 0.0;
            var relative_position_y = 1 - MLMath.Lerp(_pointsY[0], y_min, y_max);

            painter2D.MoveTo(Plot(relative_position_x, relative_position_y));

            for (int i = 0; i < _pointsY.Length; i++)
            {
                relative_position_x = MLMath.Lerp(i, x_min, x_max);
                relative_position_y = 1 - MLMath.Lerp(_pointsY[i], y_min, y_max);

                painter2D.LineTo(Plot(relative_position_x, relative_position_y));

            }

            painter2D.Stroke();
        }

        protected void GenerateLineXY(MeshGenerationContext ctx)
        {
            var painter2D = ctx.painter2D;

            painter2D.lineWidth = _lineWidth;
            painter2D.strokeColor = strokeColor;
        }


        protected void GenerateLineXYDynamic(MeshGenerationContext ctx)
        {
            var painter2D = ctx.painter2D;

            painter2D.lineWidth = _lineWidth;
            painter2D.strokeColor = strokeColor;
        }

        /// <summary>
        /// Generate the line without knowing any x value, so we assume a equal distribution of points on x and just compute the interval by pointsCount / avalaibleWidth 
        /// </summary>
        /// <param name="ctx"></param>
        protected void GenerateLineYDynamic(MeshGenerationContext ctx)
        {
            var painter2D = ctx.painter2D;

            painter2D.lineWidth = _lineWidth;
            painter2D.strokeColor = strokeColor;

            var points = _getYValuesDelegates();

            float deltaY = (int)(real_width / points.Count);

            MLMath.ColumnMinMax(points, out y_min, out y_max);

            x_min = 0;
            x_max = points.Count;

            painter2D.BeginPath();

            var relative_position_x = 0.0;
            var relative_position_y = 1 - MLMath.Lerp(points[0], y_min, y_max);

            painter2D.MoveTo(Plot(relative_position_x, relative_position_y));

            for (int i = 0; i < points.Count; i++)
            {
                relative_position_x = MLMath.Lerp(i, x_min, x_max);
                relative_position_y = 1 - MLMath.Lerp(points[i], y_min, y_max);

                painter2D.LineTo(Plot(relative_position_x, relative_position_y));

            }

            painter2D.Stroke();
        }
    }
}
