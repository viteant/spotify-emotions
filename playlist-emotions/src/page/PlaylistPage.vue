<script setup lang="ts">
import { onMounted, ref } from 'vue'
import type { Cluster } from '@/types/PlayList.ts'
import PlaylistCluster from '@/components/PlaylistCluster.vue'

const api = import.meta.env.VITE_API_URL
const clusters = ref<Cluster[]>([])
const getClusters = async () => {
  try {
    const res = await fetch(`${api}/clusters`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    clusters.value = await res.json()
    console.log(clusters.value)
  } catch (err: unknown) {
    console.error(err)
  }
}

onMounted(() => {
  getClusters()
})
</script>

<template>
  <h1 class="text-4xl text-center text-emerald-300">Suggested Playlists</h1>
  <h2 class="text-xl text-center text-emerald-100 mb-10 font-light">
    Suggested playlists with keyword clustering.
  </h2>
  <div class="p-6">
    <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 space-y-4">
      <div v-for="cluster in clusters" :key="cluster.id">
        <PlaylistCluster
          :title="cluster.name"
          :cluster-id="cluster.id"
          :description="cluster.description"
        />
      </div>
    </div>
  </div>
</template>

<style scoped></style>
