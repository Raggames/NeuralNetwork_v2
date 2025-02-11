namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    public enum OHCDTimeIntervals
    {
        OneSecond,
        FiveSeconds,
        TenSeconds,
        OneMinute,
        FiveMinute,
        HalfHour,
        Hour,
        Day,
        Week
    }

    public static class TimeIntervalExtensions
    {

        public static string Interval(OHCDTimeIntervals interval)
        {
            switch (interval)
            {
                case OHCDTimeIntervals.OneSecond:
                    return "1s";
                case OHCDTimeIntervals.FiveSeconds:
                    return "5s";
                case OHCDTimeIntervals.TenSeconds:
                    return "10s";
                case OHCDTimeIntervals.OneMinute:
                    return "1min";
                case OHCDTimeIntervals.FiveMinute:
                    return "5min";
                case OHCDTimeIntervals.HalfHour:
                    return "15min";
                case OHCDTimeIntervals.Hour:
                    return "1h";
                case OHCDTimeIntervals.Day:
                    return "1day";
                case OHCDTimeIntervals.Week:
                    return "1week";
            }

            return null;
        }
    }
}
