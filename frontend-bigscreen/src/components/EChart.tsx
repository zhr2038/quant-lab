import { GaugeChart, LineChart, PieChart, RadarChart, SankeyChart } from "echarts/charts";
import {
  GridComponent,
  GraphicComponent,
  RadarComponent,
  TooltipComponent
} from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import ReactEChartsCore from "echarts-for-react/lib/core";
import type { EChartsReactProps } from "echarts-for-react";

echarts.use([
  CanvasRenderer,
  GaugeChart,
  PieChart,
  LineChart,
  SankeyChart,
  RadarChart,
  GridComponent,
  GraphicComponent,
  RadarComponent,
  TooltipComponent
]);

type ReactEChartsProps = Omit<EChartsReactProps, "echarts">;

export function ReactECharts(props: ReactEChartsProps) {
  return <ReactEChartsCore echarts={echarts} {...props} />;
}
