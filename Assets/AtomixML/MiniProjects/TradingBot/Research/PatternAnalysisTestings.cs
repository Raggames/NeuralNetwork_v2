using Atom.MachineLearning.Core.Maths;
using Atomix.ChartBuilder;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UIElements;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    public class PatternAnalysisTestings : MonoBehaviour
    {
        [Header("Visualization")]
        [SerializeField] private VisualizationSheet _visualizationSheet;

        [SerializeField] private TradingBotManager _tradingBotManager;  


        #region Visualization

        [SerializeField, Range(0f, 1f), OnValueChanged(nameof(updateCurrentView))] private float _horizontal;
        [SerializeField, Range(0f, 1f), OnValueChanged(nameof(updateCurrentView))] private float _zoom;

        [SerializeField] private float _priceNoiseLevel = .1f;

        private List<MarketData> _currentMarketSamples;
        private void updateCurrentView()
        {
            //VisualizeDataset();
        }

        [SerializeField] private Color _bullishColor;
        [SerializeField] private Color _bearishColor;

        [Button]
        private void Test_CandlePatterns(float sigma = .02f, float threshold  =1)
        {
            int ticks = 50;

            var position = _horizontal;
            var range = _zoom;
            _currentMarketSamples = _tradingBotManager.GetMarketDatas(false);

            _visualizationSheet.Awake();
            var root = _visualizationSheet.AddPixelSizedContainer("c0", new Vector2Int(1920, 1080));
            root.style.flexDirection = new UnityEngine.UIElements.StyleEnum<UnityEngine.UIElements.FlexDirection>(UnityEngine.UIElements.FlexDirection.Column);
            root.style.flexWrap = new StyleEnum<Wrap>(StyleKeyword.Auto);

            var container = _visualizationSheet.AddContainer("c0", Color.black, new Vector2Int(100, 100), root);
            container.SetPadding(10, 10, 10, 10);

            var entity = _tradingBotManager.GenerateTradingBot_EmaScalping();
            var strategy = new MomentumScalpingStrategy();
            ((ITradingBotStrategy<TradingBotEntity>)strategy).Initialize(entity);


            int start = (int)(_currentMarketSamples.Count * position);
            int sample = (int)(range * (_currentMarketSamples.Count - start));

            var slice = _currentMarketSamples.GetRange(start, sample);
            var slice_close = new double[slice.Count, 5];
            var volume = new double[slice.Count, 2];
            int i = 0;

            var engulfing_bullish_points = new Dictionary<int, MarketData>();
            var engulfing_bearish_points = new Dictionary<int, MarketData>();

            var fractal_bullish_points = new Dictionary<int, MarketData>();
            var fractal_bearish_points = new Dictionary<int, MarketData>();
            foreach (var item in slice)
            {
                slice_close[i, 0] = i;
                slice_close[i, 1] = (double)item.Low;
                slice_close[i, 2] = (double)item.High;
                slice_close[i, 3] = (double)item.Open;
                slice_close[i, 4] = (double)item.Close;
                volume[i, 0] = i;
                volume[i, 1] = item.Open > item.Close ? -item.Volume : item.Volume;

                if(i > 3)
                {
                    if (CandlePatternFinder.BullishEngulfing(slice, i, item.Close, Convert.ToDecimal(threshold)) > 1)
                    {
                        Debug.LogError("bullish engulf at" + i);
                        engulfing_bullish_points.Add(i, item);
                    }

                    if (CandlePatternFinder.BearishEngulfing(slice, i, item.Close, Convert.ToDecimal(threshold)) > 1)
                    {
                        Debug.LogError("bullish engulf at" + i);
                        engulfing_bearish_points.Add(i, item);
                    }

                }

                i++;
            }


            var line = _visualizationSheet.Add_CandleBars(slice_close, new Vector2Int(100, 100), container);
            line.SetPadding(50, 50, 50, 50);
            line.SetTitle("Market Prices + ema");
            line.DrawAutomaticGrid();

            line.AppendScatter(ToPointMatrix(engulfing_bullish_points), _bullishColor, 4);
            line.Refresh();

            line.AppendScatter(ToPointMatrix(engulfing_bearish_points), _bearishColor, 4);
            line.Refresh();

        }


        [Button]
        private void Test_Fractals(float sigma = .02f, float threshold  =1)
        {
            int ticks = 50;

            var position = _horizontal;
            var range = _zoom;
            _currentMarketSamples = _tradingBotManager.GetMarketDatas(false);

            _visualizationSheet.Awake();
            var root = _visualizationSheet.AddPixelSizedContainer("c0", new Vector2Int(1920, 1080));
            root.style.flexDirection = new UnityEngine.UIElements.StyleEnum<UnityEngine.UIElements.FlexDirection>(UnityEngine.UIElements.FlexDirection.Column);
            root.style.flexWrap = new StyleEnum<Wrap>(StyleKeyword.Auto);

            var container = _visualizationSheet.AddContainer("c0", Color.black, new Vector2Int(100, 100), root);
            container.SetPadding(10, 10, 10, 10);

            var entity = _tradingBotManager.GenerateTradingBot_EmaScalping();
            var strategy = new MomentumScalpingStrategy();
            ((ITradingBotStrategy<TradingBotEntity>)strategy).Initialize(entity);


            int start = (int)(_currentMarketSamples.Count * position);
            int sample = (int)(range * (_currentMarketSamples.Count - start));

            var slice = _currentMarketSamples.GetRange(start, sample);
            var slice_close = new double[slice.Count, 5];
            var volume = new double[slice.Count, 2];
            int i = 0;

            var engulfing_bullish_points = new Dictionary<int, MarketData>();
            var engulfing_bearish_points = new Dictionary<int, MarketData>();

            var fractal_bullish_points = new Dictionary<int, MarketData>();
            var fractal_bearish_points = new Dictionary<int, MarketData>();
            foreach (var item in slice)
            {
                slice_close[i, 0] = i;
                slice_close[i, 1] = (double)item.Low;
                slice_close[i, 2] = (double)item.High;
                slice_close[i, 3] = (double)item.Open;
                slice_close[i, 4] = (double)item.Close;
                volume[i, 0] = i;
                volume[i, 1] = item.Open > item.Close ? -item.Volume : item.Volume;

                var fractal = DirectionalChange.FindFractals(slice, i);
                if (fractal != null)
                {
                    if (fractal.Type == DirectionalChange.FractalType.Bullish)
                        fractal_bullish_points.Add(i, item);

                    if (fractal.Type == DirectionalChange.FractalType.Bearish)
                        fractal_bearish_points.Add(i, item);
                }

                i++;
            }


            var line = _visualizationSheet.Add_CandleBars(slice_close, new Vector2Int(100, 100), container);
            line.SetPadding(50, 50, 50, 50);
            line.SetTitle("Market Prices + ema");
            line.DrawAutomaticGrid();

            line.AppendScatter(ToPointMatrix(fractal_bullish_points), _bullishColor, 4);
            line.Refresh();

            line.AppendScatter(ToPointMatrix(fractal_bearish_points), _bearishColor, 4);
            line.Refresh();

        }
        
        [Button]
        private void Test_RollingWindow(int windowsize = 3)
        {
            int ticks = 50;

            var position = _horizontal;
            var range = _zoom;
            _currentMarketSamples = _tradingBotManager.GetMarketDatas(false);

            _visualizationSheet.Awake();
            var root = _visualizationSheet.AddPixelSizedContainer("c0", new Vector2Int(1920, 1080));
            root.style.flexDirection = new UnityEngine.UIElements.StyleEnum<UnityEngine.UIElements.FlexDirection>(UnityEngine.UIElements.FlexDirection.Column);
            root.style.flexWrap = new StyleEnum<Wrap>(StyleKeyword.Auto);

            var container = _visualizationSheet.AddContainer("c0", Color.black, new Vector2Int(100, 100), root);
            container.SetPadding(10, 10, 10, 10);

            var entity = _tradingBotManager.GenerateTradingBot_EmaScalping();
            var strategy = new MomentumScalpingStrategy();
            ((ITradingBotStrategy<TradingBotEntity>)strategy).Initialize(entity);


            int start = (int)(_currentMarketSamples.Count * position);
            int sample = (int)(range * (_currentMarketSamples.Count - start));

            var slice = _currentMarketSamples.GetRange(start, sample);
            var slice_close = new double[slice.Count, 5];
            var volume = new double[slice.Count, 2];
            int i = 0;

            var rollingWindow = new RollingWindowPivots(windowsize);

            var fractal_bullish_points = new Dictionary<int, MarketData>();
            var fractal_bearish_points = new Dictionary<int, MarketData>();
            foreach (var item in slice)
            {
                slice_close[i, 0] = i;
                slice_close[i, 1] = (double)item.Low;
                slice_close[i, 2] = (double)item.High;
                slice_close[i, 3] = (double)item.Open;
                slice_close[i, 4] = (double)item.Close;
                volume[i, 0] = i;
                volume[i, 1] = item.Open > item.Close ? -item.Volume : item.Volume;

                rollingWindow.AddDataPoint(item);

                if( i > windowsize)
                {
                    int result = rollingWindow.CheckForReversal();
                    if (result == 1)
                    {
                        fractal_bullish_points.Add(i, item);
                    }
                    else if (result == -1)
                    {
                        fractal_bearish_points.Add(i, item);
                    }
                }
                
                i++;
            }


            var line = _visualizationSheet.Add_CandleBars(slice_close, new Vector2Int(100, 100), container);
            line.SetPadding(50, 50, 50, 50);
            line.SetTitle("Market Prices + ema");
            line.DrawAutomaticGrid();

            line.AppendScatter(ToPointMatrix(fractal_bullish_points), _bullishColor, 4);
            line.Refresh();

            line.AppendScatter(ToPointMatrix(fractal_bearish_points), _bearishColor, 4);
            line.Refresh();

        }


        public static double[,] ToPointMatrix(Dictionary<int, MarketData> pointsDict)
        {
            double[,] array = new double[pointsDict.Count, 2];
            int i = 0;
            foreach (var v in pointsDict)
            {
                array[i, 0] = v.Key;
                array[i, 1] = (double)v.Value.Close;
                i++;
            }
            return array;
        }

        [Button]
        private void Test_EmaStrategy()
        {
            int ticks = 50;

            var position = _horizontal;
            var range = _zoom;
            _currentMarketSamples = _tradingBotManager.GetMarketDatas(false);

            _visualizationSheet.Awake();
            var root = _visualizationSheet.AddPixelSizedContainer("c0", new Vector2Int(1920, 1080));
            root.style.flexDirection = new UnityEngine.UIElements.StyleEnum<UnityEngine.UIElements.FlexDirection>(UnityEngine.UIElements.FlexDirection.Column);
            root.style.flexWrap = new StyleEnum<Wrap>(StyleKeyword.Auto);

            var container = _visualizationSheet.AddContainer("c0", Color.black, new Vector2Int(100, 100), root);
            container.SetPadding(10, 10, 10, 10);

            var entity = _tradingBotManager.GenerateTradingBot_EmaScalping();
            var strategy = new MomentumScalpingStrategy();
            ((ITradingBotStrategy<TradingBotEntity>)strategy).Initialize(entity);


            int start = (int)(_currentMarketSamples.Count * position);
            int sample = (int)(range * (_currentMarketSamples.Count - start));

            var slice = _currentMarketSamples.GetRange(start, sample);
            var slice_close = new double[slice.Count, 5];
            var volume = new double[slice.Count, 2];
            int i = 0;

            bool onPosition = false;
            Dictionary<Color, double[,]> strategyResults = new Dictionary<Color, double[,]>();
            var entries = new List<int>();
            var entriesPrices = new List<float>();
            var sentries = new List<int>();
            var sentriesPrices = new List<float>();

            var exits = new List<int>();
            var exitsPrices = new List<float>();
            var entriesColors = new List<Color>();

            foreach (var item in slice)
            {
                slice_close[i, 0] = i;
                slice_close[i, 1] = (double)item.Low;
                slice_close[i, 2] = (double)item.High;
                slice_close[i, 3] = (double)item.Open;
                slice_close[i, 4] = (double)item.Close;
                volume[i, 0] = i;
                volume[i, 1] = item.Open > item.Close ? -item.Volume : item.Volume;

                strategy.OnOHLCUpdate(item);
                for (int j = 0; j < ticks; ++j)
                {
                    var price = PriceUtils.GenerateMovementPrice(item.Open, item.Low, item.High, item.Close, .1f, j, ticks);
                    strategy.OnTick(item, price);

                    if (!onPosition)
                    {
                        var pos = strategy.CheckEntryConditions(price);
                        switch (pos)
                        {
                            case PositionTypes.Long_Buy:
                                onPosition = true;
                                entries.Add(i);
                                entriesPrices.Add((float)price);
                                break;
                            case PositionTypes.Short_Sell:
                                onPosition = true;
                                sentries.Add(i);
                                sentriesPrices.Add((float)price);

                                break;
                        }

                    }
                    else
                    {
                        if (strategy.CheckExitConditions(price))
                        {
                            exits.Add(i);
                            onPosition = false;
                        }
                    }

                }

                i++;
            }

            var line = _visualizationSheet.Add_CandleBars(slice_close, new Vector2Int(100, 100), container);
            line.SetPadding(50, 50, 50, 50);
            line.SetTitle("Market Prices + ema");
            line.DrawAutomaticGrid();

            var longs = new double[entries.Count, 2];
            for (int k = 0; k < entries.Count; k++)
            {
                longs[k, 0] = entries[k];
                longs[k, 1] = entriesPrices[k];
            }

            line.AppendScatter(longs, Color.black, 2);

            var shorts = new double[sentries.Count, 2];
            for (int k = 0; k < sentries.Count; k++)
            {
                shorts[k, 0] = sentries[k];
                shorts[k, 1] = sentriesPrices[k];
            }

            line.AppendScatter(shorts, Color.black, 2);

            var exitss = new double[exits.Count, 2];
            for (int k = 0; k < sentries.Count; k++)
            {
                exitss[k, 0] = exits[k];
                exitss[k, 1] = exitsPrices[k];
            }

            line.AppendScatter(shorts, Color.black, 2);
            line.Refresh();
        }

        [Button]
        private void VisualizeDataset()
        {
            if (_currentMarketSamples == null)
                _currentMarketSamples = _tradingBotManager.GetMarketDatas(false);

            _visualizationSheet.Awake();
            var root = _visualizationSheet.AddPixelSizedContainer("c0", new Vector2Int(1920, 1080));
            root.style.flexDirection = new UnityEngine.UIElements.StyleEnum<UnityEngine.UIElements.FlexDirection>(UnityEngine.UIElements.FlexDirection.Column);
            root.style.flexWrap = new StyleEnum<Wrap>(StyleKeyword.Auto);

            var container = _visualizationSheet.AddContainer("c0", Color.black, new Vector2Int(100, 50), root);
            container.SetPadding(10, 10, 10, 10);

            int start = (int)(_currentMarketSamples.Count * _horizontal);
            int sample = (int)(_zoom * (_currentMarketSamples.Count - start));

            var slice = _currentMarketSamples.GetRange(start, sample);
            var slice_close = new double[slice.Count, 5];
            var volume = new double[slice.Count, 2];
            int i = 0;
            foreach (var item in slice)
            {
                slice_close[i, 0] = i;
                slice_close[i, 1] = (double)item.Low;
                slice_close[i, 2] = (double)item.High;
                slice_close[i, 3] = (double)item.Open;
                slice_close[i, 4] = (double)item.Close;
                volume[i, 0] = i;
                volume[i, 1] = item.Open > item.Close ? -item.Volume : item.Volume;
                i++;
            }

            var tech = new TechnicalAnalysis();
            tech.Initialize();

            var line = _visualizationSheet.Add_CandleBars(slice_close, new Vector2Int(100, 100), container);
            line.SetPadding(50, 50, 50, 50);
            line.SetTitle("Market Prices + ema");
            line.DrawAutomaticGrid();

            // line.AppendVerticalBar(volume);

            double[,] ema5 = new double[slice.Count, 2];
            i = 0;
            foreach (var item in slice)
            {
                tech.Update(item);
                ema5[i, 0] = i;
                ema5[i, 1] = Convert.ToDouble(tech.ema5.current);
                i++;
            }

            line.AppendLine(ema5, Color.green, 1.5f);

            double[,] ema10 = new double[slice.Count, 2];
            i = 0;
            foreach (var item in slice)
            {
                tech.Update(item);
                ema10[i, 0] = i;
                ema10[i, 1] = Convert.ToDouble(tech.ema10.current);
                i++;
            }
            line.AppendLine(ema10, Color.red, 1.5f);


            var suppRes = new PivotPoint();
            i = 0;

            foreach (var item in slice)
            {
                suppRes.Compute(item.High, item.Low, item.Close);
            }

            double[,] support = new double[,]
            {
                { 0, (double)suppRes.Support1 },
                { slice.Count,  (double)suppRes.Support1 }
            };
            double[,] res = new double[,]
            {
                { 0, (double)suppRes.Resistance1 },
                { slice.Count,  (double)suppRes.Resistance1 }
            };
            line.AppendLine(support, Color.blue, 1.5f);
            line.AppendLine(res, Color.green, 1.5f);

            line.Refresh();



            var container2 = _visualizationSheet.AddContainer("c1", Color.black, new Vector2Int(100, 50), root);
            container2.SetPadding(10, 10, 10, 10);

            double[,] rsi = new double[slice.Count, 2];
            i = 0;
            foreach (var item in slice)
            {
                tech.Update(item);
                rsi[i, 0] = i;
                rsi[i, 1] = Convert.ToDouble(tech.rsi.current);
                i++;
            }
            var line2 = _visualizationSheet.Add_SimpleLine(rsi, 2, new Vector2Int(100, 100), container2);
            line2.strokeColor = Color.yellow;
            line2.SetPadding(50, 50, 50, 50);
            line2.SetTitle("RSI");
            line2.DrawAutomaticGrid();


            double[,] cmf = new double[slice.Count, 2];
            i = 0;
            foreach (var item in slice)
            {
                tech.Update(item);
                cmf[i, 0] = i;
                cmf[i, 1] = Convert.ToDouble(tech.cmf.current);
                i++;
            }

            var line3 = _visualizationSheet.Add_SimpleLine(rsi, 2, new Vector2Int(100, 100), container2);
            line3.strokeColor = Color.magenta;
            line3.SetPadding(50, 50, 50, 50);
            line3.SetTitle("CMI");
            line3.DrawAutomaticGrid();

            var line4 = _visualizationSheet.Add_PositiveNegativeVerticalBars(volume, new Vector2Int(100, 100), container2);
            line4.strokeColor = Color.magenta;
            line4.SetPadding(50, 50, 50, 50);
            line4.SetTitle("Volume");
            line4.DrawAutomaticGrid();


            /* var line2 = _visualizationSheet.Add_SimpleLine(rsi, 2, new Vector2Int(100, 100), container);
             line2.backgroundColor = new Color(0, 0, 0, 0);*/
        }

        [Button]
        private void ShowGaussianPrice(int pointsCount = 25)
        {
            _visualizationSheet.Awake();

            var root = _visualizationSheet.AddPixelSizedContainer("c0", new Vector2Int(1000, 1000));
            root.style.flexDirection = new UnityEngine.UIElements.StyleEnum<UnityEngine.UIElements.FlexDirection>(UnityEngine.UIElements.FlexDirection.Row);
            root.style.flexWrap = new StyleEnum<Wrap>(StyleKeyword.Auto);

            var container = _visualizationSheet.AddContainer("c0", Color.black, new Vector2Int(100, 100), root);
            container.SetPadding(10, 10, 10, 10);

            var datas = _tradingBotManager.GetMarketDatas(false);
            var price = datas[MLRandom.Shared.Range(0, datas.Count)];
            var points = new double[pointsCount, 2];
            for (int i = 0; i < points.GetLength(0); i++)
            {
                points[i, 0] = i;
                points[i, 1] = (double)PriceUtils.GenerateMovementPrice(price.Open, price.Low, price.High, price.Close, _priceNoiseLevel, i, pointsCount);
            }
            var line = _visualizationSheet.Add_SimpleLine(points, 2, new Vector2Int(100, 100), container);
            line.strokeColor = Color.yellow;
            line.lineWidth = 3;
            line.SetPadding(50, 50, 50, 50);
            line.SetTitle("Prices : " + price.Volume);
            line.DrawAutomaticGrid();

            line.AppendLine(new double[,] { { 0, (double)price.Open }, { pointsCount, (double)price.Close } }, Color.red, 3);
            line.Refresh();
        }
        #endregion
    }
}
