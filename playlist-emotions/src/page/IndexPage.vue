<script setup lang="ts">
import { onMounted, ref } from 'vue'
import type {
  EmotionDictionaryEntry,
  EmotionScores,
  NormalizeEmotion,
} from '@/types/PlayListEmotions.ts'
import ChartNormalizeEmotions from '@/components/ChartNormalizeEmotions.vue'
import GlassCard from '@/components/ui/GlassCard.vue'
import EmotionsDetails from '@/components/EmotionsDetails.vue'
import EmotionsTable from '@/components/EmotionsTable.vue'

const api = import.meta.env.VITE_API_URL
const playlistNormalizeEmotions = ref<NormalizeEmotion[]>([])
const playlistEmotions = ref<EmotionScores>({})
const emotionDictionary = ref<EmotionDictionaryEntry[]>([])


const getDataEmotions = async () => {
  try {
    const res = await fetch(`${api}/playlist-emotions`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const response = await res.json()
    playlistNormalizeEmotions.value = response.emotions
  } catch (err: unknown) {
    console.error(err)
  }
}
const getAllDataEmotions = async () => {
  try {
    const res = await fetch(`${api}/playlist-emotions/all`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const response = await res.json()
    playlistEmotions.value = response.emotions
    emotionDictionary.value = response.dictionary
  } catch (err: unknown) {
    console.error(err)
  }
}

onMounted(() => {
  getDataEmotions()
  getAllDataEmotions()
})
</script>
<template>
  <h1 class="text-4xl text-center text-emerald-300">My Playlist Emotions</h1>
  <h2 class="text-xl text-center text-emerald-100 mb-10 font-light">Discover the emotional journey hidden within your playlists through detailed analysis of moods, feelings, and song-driven sentiments</h2>
  <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 mt-5 gap-5">
    <div class="col-span-1">
      <GlassCard title="General Emotions">
        <ChartNormalizeEmotions
          v-if="playlistNormalizeEmotions.length > 0"
          :emotions="playlistNormalizeEmotions"
        />
      </GlassCard>
    </div>
    <div class="col-span-1 lg:col-span-2">
      <GlassCard>
        <div class="grid grid-cols-1 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 h-full">
          <EmotionsDetails :emotions="playlistEmotions" :dictionary="emotionDictionary" />
        </div>
      </GlassCard>
    </div>
    <div class="md:col-span-2 lg:col-span-3">
      <EmotionsTable />
    </div>
  </div>
</template>

<style scoped></style>
