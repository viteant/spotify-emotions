import type { NormalizeEmotion } from '@/types/PlayListEmotions.ts'
import type { DoughnutChart } from '@/types/Charts.ts'

const normalizeEmotionDictionary: Map<string, string[]> = new Map<string, string[]>([
  ['FP', ['Full Positive', '#818cf8']],
  ['MP', ['Positive', '#22d3ee']],
  ['N', ['Neutral', '#34d399']],
  ['MN', ['Negative', '#fbbf24']],
  ['FN', ['Full Negative', '#fb7185']],
])

export const useChartBubble = () => ({
  AdapterNormalizeBubbleChart: (normalizeEmotions: NormalizeEmotion[]): DoughnutChart => {
    const colors: string[] = []
    const data: number[] = []
    const labels: string[] = []

    for (const key of normalizeEmotionDictionary.keys()) {
      const tuple = normalizeEmotionDictionary.get(key); // [label, color] | undefined
      if (!tuple) continue;

      const [label, color] = tuple;
      labels.push(label);
      colors.push(color);

      const value = normalizeEmotions.find(el => el.normalize === key)?.sum_avg ?? 0;
      data.push(value); // ← sin coma final
    }

    return {
      labels: labels,
      datasets: [
        {
          backgroundColor: colors,
          data: data,
        },
      ],
    }
  },
})
