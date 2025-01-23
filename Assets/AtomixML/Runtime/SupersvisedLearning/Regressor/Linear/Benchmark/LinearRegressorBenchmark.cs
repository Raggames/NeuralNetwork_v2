using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atomix.ChartBuilder;
using Atomix.ChartBuilder.VisualElements;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

namespace Atom.MachineLearning.Supervised.Regressor.Linear
{
    public class LinearRegressorBenchmark : MonoBehaviour
    {
        [SerializeField] private VisualizationSheet _visualizationSheet;
        [SerializeField] private LinearRegressorModel _model;
        [SerializeField] private NVector[] _dataset;

        [Button]
        private async void TestLinearRegressor()
        {
            _model.Weights = new NVector(1);
            _model.SetScoringMetricFunction(MLMetricFunctions.MMSE);


            var t_datas = _dataset.ColumnToVector(_dataset[0].length - 1);
            var x_datas = _dataset.RemoveColumn(1);

            var result = await _model.Fit(x_datas, t_datas.Data);

            var positions = new NVector[x_datas.Length];    
            for(int i = 0; i < _dataset.Length; i++)
            {
                var y = _model.Predict(x_datas[i]);
                positions[i] = new NVector(x_datas[i][0], y);
            }

            var scatter = DrawBaseDatasetGraph();
            var line = _visualizationSheet.Add_SimpleLine(positions.ToDoubleMatrix(), 2, new Vector2Int(100, 100), scatter);
            line.SetAbsolutePosition();
            line.SetDimensions(LengthUnit.Pixel, (int)scatter.resolvedStyle.width, (int)scatter.resolvedStyle.height);

            line.DrawAutomaticGrid();
            line.gridColor = new Color(0, 0, 0, 0);
            line.lineWidth = 2;

            line.SetPadding(50, 50, 50, 50);
        }


        [Button]
        private void TestScore()
        {
            Debug.Log(_model.ScoreSynchronously());
        }


        private float _a = 0;
        private float _b = 0;

        [Button]
        private void CreateSamples(int points_count = 100, float a = .5f, float b = .5f, float noise = .05f)
        {
            _a = a; 
            _b = b;

            MLRandom.SeedShared(DateTime.Now.Millisecond);

            _dataset = new NVector[points_count + 1];
            _dataset[0] = new NVector(0, 0);
            for (int i = 1; i < points_count + 1; i++)
            {
                var x = MLRandom.Shared.Range(0.0, 10.0);
                var y = a * x + b + MLRandom.Shared.Range(-noise, noise);

                _dataset[i] = new NVector(x, y);
            }

            var root = DrawBaseDatasetGraph();
        }

        private ChartBuilderElement DrawBaseDatasetGraph()
        {
            _visualizationSheet.Awake();

            var root = _visualizationSheet.AddPixelSizedContainer("c0", new Vector2Int(500, 500));
            root.style.flexDirection = new UnityEngine.UIElements.StyleEnum<UnityEngine.UIElements.FlexDirection>(UnityEngine.UIElements.FlexDirection.Row);

            var container = _visualizationSheet.AddContainer("c0", Color.black, new Vector2Int(100, 100), root);
            container.SetPadding(10, 10, 10, 10);

            //var scatter = _visualizationSheet.Add_Scatter(array.ToDoubleMatrix(), new Vector2Int(100, 100), container);
            var scatter = _visualizationSheet.Add_Scatter(_dataset.ToDoubleMatrix(), new Vector2Int(100, 100), container);

            scatter.gridColor = Color.black;
            scatter.lineWidth = 1;

            scatter.SetPadding(50, 50, 50, 50);
            scatter.SetTitle("Random points");
            scatter.DrawAutomaticGrid();


            var vec = new NVector(1, _a, 0);
            var orth = NVector.Cross(vec, new NVector(0, 0, -1));

            Debug.Log(orth);

            var x_axis = new NVector(vec.x, vec.y).normalized;
            var y_axis = new NVector(orth.x, orth.y).normalized;

            var transformed = new NVector[_dataset.Length];
            for (int i = 0; i < _dataset.Length; ++i)
            {
                var transformed_point_x = NVector.Dot(_dataset[i], x_axis) / x_axis.sqrdMagnitude;
                var transformed_point_y = (NVector.Dot(_dataset[i], y_axis) / y_axis.sqrdMagnitude) - _b;

                transformed[i] = new NVector(transformed_point_x, transformed_point_y);
            }

            var scatter_cleaned = _visualizationSheet.Add_Scatter(transformed.ToDoubleMatrix(), new Vector2Int(100, 100), container);
            scatter_cleaned.lineWidth = 1;
            scatter_cleaned.SetPadding(50, 50, 50, 50);
            scatter_cleaned.SetTitle("Transformed points (best-fit)");
            scatter_cleaned.DrawAutomaticGrid();

            return scatter;

        }

        
    }
}
