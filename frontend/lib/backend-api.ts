export type ApiRecord = Record<string, unknown>

export type BackendState = {
  overview: ApiRecord | ApiRecord[] | null
  suspicious: ApiRecord[]
  models: ApiRecord[]
  network: ApiRecord | ApiRecord[] | null
  score: ApiRecord | null
}

const configuredBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL?.trim()
const baseUrl = configuredBaseUrl ? configuredBaseUrl.replace(/\/$/, '') : ''

export async function fetchBackend<T>(path: string, init?: RequestInit): Promise<T> {
  const targetUrl = baseUrl ? `${baseUrl}${path}` : path
  const response = await fetch(targetUrl, { ...init, headers: { Accept: 'application/json', ...init?.headers } })
  if (!response.ok) throw new Error(`Backend request failed (${response.status})`)
  return response.json() as Promise<T>
}

export function value(record: ApiRecord | null | undefined, keys: string[]): unknown {
  if (!record) return undefined
  return keys.map((key) => record[key]).find((item) => item !== undefined && item !== null)
}

export function numberValue(record: ApiRecord | null | undefined, keys: string[]): number | null {
  const item = value(record, keys)
  const parsed = typeof item === 'number' ? item : Number(item)
  return Number.isFinite(parsed) ? parsed : null
}

export function records(payload: ApiRecord | ApiRecord[] | null | undefined): ApiRecord[] {
  if (Array.isArray(payload)) return payload
  if (!payload) return []
  const nested = value(payload, ['data', 'items', 'results', 'transactions', 'nodes', 'models', 'records'])
  return Array.isArray(nested) ? nested : [payload]
}
