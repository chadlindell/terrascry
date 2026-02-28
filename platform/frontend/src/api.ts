/** Typed API client for the TERRASCRY backend.
 *
 * All interfaces mirror the backend Pydantic models exactly.
 * API functions return typed promises and throw {@link ApiError} on failure.
 */

// --- Scenario types (mirrors backend api_models.py) ---

/** Terrain extents and elevation for a scenario. */
export interface TerrainSummary {
  x_extent: number[]
  y_extent: number[]
  surface_elevation: number
}

/** Summary of a buried object (name, type, position, radius). */
export interface ObjectSummary {
  name: string
  object_type: string
  position: number[]
  radius: number
}

/** Compact scenario representation returned by the list endpoint. */
export interface ScenarioSummary {
  name: string
  file_name: string
  description: string
  object_count: number
  terrain: TerrainSummary
}

/** Full scenario with objects, earth field, and raw JSON. */
export interface ScenarioDetail extends ScenarioSummary {
  objects: ObjectSummary[]
  earth_field: number[]
  metadata: Record<string, unknown>
  raw: Record<string, unknown>
}

// --- Dataset types (mirrors backend dataset.py) ---

/** Regular grid of gradient values (flat row-major) for 2D heatmap rendering. */
export interface GridData {
  rows: number
  cols: number
  x_min: number
  y_min: number
  dx: number
  dy: number
  values: number[]
  unit: string
}

/** Single survey reading: position and vertical gradient in nanoTesla. */
export interface SurveyPoint {
  x: number
  y: number
  gradient_nt: number
}

/** Metadata for a stored simulation dataset. */
export interface DatasetMetadata {
  id: string
  scenario_name: string
  created_at: string
  params: Record<string, unknown>
}

/** Complete simulation output: metadata + grid + survey points. */
export interface Dataset {
  metadata: DatasetMetadata
  grid_data: GridData
  survey_points: SurveyPoint[]
}

/** Request parameters for triggering a survey simulation. */
export interface SimulateRequest {
  scenario_name: string
  line_spacing?: number
  sample_spacing?: number
  resolution?: number
}

// --- API functions ---

class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
  ) {
    super(detail)
    this.name = 'ApiError'
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new ApiError(res.status, body.detail ?? res.statusText)
  }
  return res.json()
}

/** Fetch all scenarios from the backend. */
export async function fetchScenarios(): Promise<ScenarioSummary[]> {
  const res = await fetch('/api/scenarios')
  return handleResponse(res)
}

/** Fetch full detail for a single scenario by file name. */
export async function fetchScenario(name: string): Promise<ScenarioDetail> {
  const res = await fetch(`/api/scenarios/${encodeURIComponent(name)}`)
  return handleResponse(res)
}

/** Trigger a survey simulation and return the resulting dataset. */
export async function simulateSurvey(request: SimulateRequest): Promise<Dataset> {
  const res = await fetch('/api/surveys/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  return handleResponse(res)
}

/** Fetch metadata for all stored datasets, newest first. */
export async function fetchDatasets(): Promise<DatasetMetadata[]> {
  const res = await fetch('/api/datasets')
  return handleResponse(res)
}

/** Delete a dataset by ID. */
export async function deleteDataset(id: string): Promise<void> {
  const res = await fetch(`/api/datasets/${id}`, { method: 'DELETE' })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new ApiError(res.status, body.detail ?? res.statusText)
  }
}

/** Upload a CSV file and return the created dataset. */
export async function uploadDataset(file: File): Promise<Dataset> {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch('/api/imports/upload', {
    method: 'POST',
    body: formData,
  })
  return handleResponse(res)
}

// --- Streaming API ---

/** Start streaming data (simulation or MQTT mode). */
export async function startStream(scenarioName: string): Promise<void> {
  const res = await fetch('/api/streaming/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scenario_name: scenarioName }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new ApiError(res.status, body.detail ?? res.statusText)
  }
}

/** Stop streaming. */
export async function stopStream(): Promise<void> {
  const res = await fetch('/api/streaming/stop', { method: 'POST' })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new ApiError(res.status, body.detail ?? res.statusText)
  }
}

// --- Anomaly API ---

/** Anomaly cell returned by the anomaly detection endpoint. */
export interface AnomalyCell {
  x: number
  y: number
  gradient_nt: number
  residual_nt: number
  sigma: number
}

/** Fetch detected anomalies for a dataset. */
export async function fetchAnomalies(
  id: string,
  thresholdSigma: number = 3.0,
): Promise<AnomalyCell[]> {
  const res = await fetch(
    `/api/datasets/${id}/anomalies?threshold_sigma=${thresholdSigma}`,
  )
  return handleResponse(res)
}

// --- Binary grid transfer ---

/** Fetch grid data as binary Float32Array for faster transfer. */
export async function fetchDatasetBinary(id: string): Promise<GridData> {
  const res = await fetch(`/api/datasets/${id}/binary`)
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new ApiError(res.status, body.detail ?? res.statusText)
  }

  const rows = parseInt(res.headers.get('X-Grid-Rows') ?? '0', 10)
  const cols = parseInt(res.headers.get('X-Grid-Cols') ?? '0', 10)
  const x_min = parseFloat(res.headers.get('X-Grid-Xmin') ?? '0')
  const y_min = parseFloat(res.headers.get('X-Grid-Ymin') ?? '0')
  const dx = parseFloat(res.headers.get('X-Grid-Dx') ?? '0')
  const dy = parseFloat(res.headers.get('X-Grid-Dy') ?? '0')
  const unit = res.headers.get('X-Grid-Unit') ?? 'nT'

  const buffer = await res.arrayBuffer()
  const values = Array.from(new Float32Array(buffer))

  return { rows, cols, x_min, y_min, dx, dy, values, unit }
}
