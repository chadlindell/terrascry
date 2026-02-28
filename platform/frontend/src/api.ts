/** Typed API client for the TERRASCRY backend. */

// --- Scenario types (mirrors backend api_models.py) ---

export interface TerrainSummary {
  x_extent: number[]
  y_extent: number[]
  surface_elevation: number
}

export interface ObjectSummary {
  name: string
  object_type: string
  position: number[]
  radius: number
}

export interface ScenarioSummary {
  name: string
  file_name: string
  description: string
  object_count: number
  terrain: TerrainSummary
}

export interface ScenarioDetail extends ScenarioSummary {
  objects: ObjectSummary[]
  earth_field: number[]
  metadata: Record<string, unknown>
  raw: Record<string, unknown>
}

// --- Dataset types (mirrors backend dataset.py) ---

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

export interface SurveyPoint {
  x: number
  y: number
  gradient_nt: number
}

export interface DatasetMetadata {
  id: string
  scenario_name: string
  created_at: string
  params: Record<string, unknown>
}

export interface Dataset {
  metadata: DatasetMetadata
  grid_data: GridData
  survey_points: SurveyPoint[]
}

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

export async function fetchScenarios(): Promise<ScenarioSummary[]> {
  const res = await fetch('/api/scenarios')
  return handleResponse(res)
}

export async function fetchScenario(name: string): Promise<ScenarioDetail> {
  const res = await fetch(`/api/scenarios/${encodeURIComponent(name)}`)
  return handleResponse(res)
}

export async function simulateSurvey(request: SimulateRequest): Promise<Dataset> {
  const res = await fetch('/api/surveys/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  return handleResponse(res)
}
