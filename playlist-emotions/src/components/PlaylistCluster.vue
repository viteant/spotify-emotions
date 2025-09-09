<script setup lang="ts">
import { onMounted, ref } from 'vue'
import type { Track, ClusterPlaylistResponse, ClusterPlaylistBody } from '@/types/PlayList.ts'
import PlaylistCard from '@/components/ui/PlaylistCard.vue'

const props = defineProps<{
  clusterId: number
  title: string
  description: string
}>()
const api = import.meta.env.VITE_API_URL
const tracks = ref<Track[]>([])
const getTrack = async () => {
  try {
    const res = await fetch(`${api}/playlist-cluster/${props.clusterId}`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    tracks.value = await res.json()
    //console.log(tracks.value)
  } catch (err: unknown) {
    console.error(err)
  }
}

onMounted(() => {
  getTrack()
})
</script>

<template>
  <PlaylistCard
    :title="title"
    :tracks="tracks"
    :description="description"
    :cluster_id="clusterId"
  />
</template>

<style scoped></style>
