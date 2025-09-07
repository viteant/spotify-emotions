<script setup lang="ts">
//@ts-expect-error
import Vue3Datatable from '@bhplugin/vue3-datatable'
import '@bhplugin/vue3-datatable/dist/style.css'
import { onMounted, ref } from 'vue'
import type {
  Emotion,
  EmotionEntry,
  Filters,
  SortBy,
  SortDir,
  TrackItem,
  TracksResponse,
} from '@/types/TracksEmotions.ts'
import TrackFilter from '@/components/ui/TrackFilter.vue'

const renderPorc = (value: string) => {
  return `${Number.parseInt(value).toFixed(2)} %`
}

function getMaxEmotion(emotions: EmotionEntry[]): string {
  if (!emotions || emotions.length === 0) {
    return ''
  }
  const emotion = emotions.reduce((max, current) =>
    current.score > max.score ? current : max,
  ).label

  const emotionFind = emotionsDictionary.value?.find((el: Emotion) => el.emotion == emotion)
  if (emotionFind === undefined) {
    return ''
  }
  return `${emotionFind.emotion.toUpperCase()} (${emotionFind.emoji})`
}

const cols = ref([
  { field: 'track', title: 'Track', sort: true, width: '250px' },
  { field: 'artist', title: 'Artist', sort: true, width: '250px' },
  { field: 'album', title: 'Album', sort: true, width: '250px' },
  {
    field: 'main_emotion',
    title: 'Main Emotion',
    sort: false,
    cellRenderer: (value: any) => getMaxEmotion(JSON.parse(value.emotions)),
  },
  {
    field: 'full_positive',
    title: 'Positive++',
    sort: true,
    cellRenderer: (value: any) => renderPorc(value.full_positive),
  },
  {
    field: 'positive',
    title: 'Positive',
    sort: true,
    cellRenderer: (value: any) => renderPorc(value.positive),
  },
  {
    field: 'neutral',
    title: 'Neutral',
    sort: true,
    cellRenderer: (value: any) => renderPorc(value.neutral),
  },
  {
    field: 'negative',
    title: 'Negative',
    sort: true,
    cellRenderer: (value: any) => renderPorc(value.negative),
  },
  {
    field: 'full_negative',
    title: 'Negative++',
    sort: true,
    cellRenderer: (value: any) => renderPorc(value.full_negative),
  },
])

const rows = ref<TrackItem[]>([])
const page = ref<number>()
const pageSize = ref<number>(10)
const total = ref<number>(0)
const sortBy = ref<SortBy>('track')
const sortDir = ref<SortDir>('asc')
const loading = ref<boolean>(false)
const emotionsDictionary = ref<Emotion[]>([])

interface GetDataParams {
  page?: number // default 1
  page_size?: number // default 10
  track?: string | null
  artist?: string | null
  album?: string | null
  sort_by?: SortBy // default "track"
  sort_dir?: SortDir // default "asc"
}

const api = import.meta.env.VITE_API_URL

const getEmotionsDictionary = async () => {
  const url = `${api}/dictionary/emotions`

  const res = await fetch(url, {
    method: 'GET',
    headers: { Accept: 'application/json' },
  })

  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`Request failed: ${res.status} ${res.statusText} ${text}`.trim())
  }

  const emotions = await res.json()
  emotionsDictionary.value = emotions
}
async function getData(params: GetDataParams = {}): Promise<TracksResponse> {
  const {
    page = 1,
    page_size = 10,
    track,
    artist,
    album,
    sort_by = 'track',
    sort_dir = 'asc',
  } = params

  const qs = new URLSearchParams()
  qs.set('page', String(page))
  qs.set('page_size', String(page_size))
  qs.set('sort_by', sort_by as SortBy)
  qs.set('sort_dir', sort_dir as SortDir)
  if (track && track.trim()) qs.set('track', track.trim())
  if (artist && artist.trim()) qs.set('artist', artist.trim())
  if (album && album.trim()) qs.set('album', album.trim())

  const url = `${api}/emotion-tracks?${qs.toString()}`

  const res = await fetch(url, {
    method: 'GET',
    headers: { Accept: 'application/json' },
  })

  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`Request failed: ${res.status} ${res.statusText} ${text}`.trim())
  }

  return (await res.json()) as TracksResponse
}
const fetchData = (params: GetDataParams) => {
  loading.value = true
  getData(params)
    .then((data) => {
      rows.value = data.items
      page.value = data.page
      pageSize.value = data.page_size
      total.value = data.total
      sortBy.value = data.sort_by
      sortDir.value = data.sort_dir
    })
    .finally(() => (loading.value = false))
}
const TableChange = (data: any) => {
  fetchData({
    page: data.current_page,
    page_size: data.pagesize,
    sort_by: data.sort_column,
    sort_dir: data.sort_direction,
  })
}
const TableFilter = (filters: Filters) => {
  fetchData({
    page: page.value,
    page_size: pageSize.value,
    sort_by: sortBy.value,
    sort_dir: sortDir.value,
    track: filters.track,
    artist: filters.artist,
    album: filters.album,
  })
}

onMounted(async () => {
  loading.value = true
  await getEmotionsDictionary()

  getData()
    .then((data) => {
      rows.value = data.items
      page.value = data.page
      pageSize.value = data.page_size
      total.value = data.total
      sortBy.value = data.sort_by
      sortDir.value = data.sort_dir
    })
    .finally(() => {
      loading.value = false
    })
})
</script>

<template>
  <div class="flex flex-row justify-end mb-2">
    <TrackFilter @apply="TableFilter" />
  </div>
  <vue3-datatable
    :rows="rows"
    :columns="cols"
    :loading="loading"
    :isServerMode="true"
    :sortable="true"
    :totalRows="total"
    :pageSize="pageSize"
    skin="bh-table-hover"
    class="glass"
    @change="TableChange"
  >
  </vue3-datatable>
</template>

<style scoped></style>
