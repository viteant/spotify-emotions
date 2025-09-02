<script setup lang="ts">
import type { NormalizeEmotion } from '@/types/PlayListEmotions.ts'
import { useChartBubble } from '@/composables/useChartBubble.ts'
import { computed } from 'vue'
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js'
import { Doughnut } from 'vue-chartjs'

ChartJS.register(ArcElement, Tooltip, Legend)

const props = defineProps<{
  emotions: NormalizeEmotion[]
}>()

const { AdapterNormalizeBubbleChart } = useChartBubble()

const dataEmotions = computed(() => AdapterNormalizeBubbleChart(props.emotions))
const options = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      display: true,
      position: 'left',   // ← move legend to the left
      align: 'center',
      labels: {
        usePointStyle: true,
        boxWidth: 10,
        padding: 16,
        color: '#9CA3AF', // optional: slate-400
      },
    },
    tooltip: {
      enabled: true,
    },
  }
}
</script>

<template>
  <Doughnut :data="dataEmotions" :options="options as any" />
</template>
