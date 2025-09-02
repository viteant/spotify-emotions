<!-- components/CompactEmotionBars.vue -->
<template>
  <div class="grid grid-cols-1 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
    <div
      v-for="entry in sorted"
      :key="entry.emotion"
      class="flex items-center gap-2"
    >
      <!-- Emoji -->
      <span class="text-lg w-6 text-center">{{ entry.emoji }}</span>

      <!-- Label + Bar -->
      <div class="flex-1">
        <div class="flex justify-between text-xs text-slate-300 mb-1">
          <span>{{ entry.emotion }}</span>
          <span>{{ entry.value.toFixed(2) }}%</span>
        </div>
        <div class="w-full bg-slate-700 rounded-full h-2">
          <div
            class="h-2 rounded-full bg-gradient-to-r from-emerald-400 to-emerald-600"
            :style="{ width: Math.min(entry.value, 100) + '%' }"
          ></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";

type EmotionScore = Record<string, number>;
interface EmotionDictionaryEntry {
  emotion: string;
  emoji: string;
}

const props = defineProps<{
  emotions: EmotionScore;
  dictionary: EmotionDictionaryEntry[];
}>();

// Merge dictionary with values, then sort by value desc
const sorted = computed(() => {
  const entries = props.dictionary.map((d) => ({
    emotion: d.emotion,
    emoji: d.emoji,
    value: props.emotions[d.emotion] ?? 0,
  }));

  return entries.sort((a, b) => b.value - a.value);
});
</script>
