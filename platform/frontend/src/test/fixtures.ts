import type {
  Dataset,
  DatasetMetadata,
  GridData,
  ScenarioDetail,
  ScenarioSummary,
  SurveyPoint,
} from '../api'

export const TEST_GRID: GridData = {
  rows: 3,
  cols: 4,
  x_min: 0.0,
  y_min: 0.0,
  dx: 1.0,
  dy: 1.0,
  values: [
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0,
  ],
  unit: 'nT',
}

export const TEST_SURVEY_POINTS: SurveyPoint[] = [
  { x: 0.0, y: 0.0, gradient_nt: 1.5 },
  { x: 1.0, y: 0.0, gradient_nt: 3.2 },
  { x: 2.0, y: 0.0, gradient_nt: -0.8 },
]

export const TEST_METADATA: DatasetMetadata = {
  id: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
  scenario_name: 'test-scenario',
  created_at: '2026-01-15T12:00:00Z',
  params: { resolution: 1.0, line_spacing: 1.0 },
}

export const TEST_DATASET: Dataset = {
  metadata: TEST_METADATA,
  grid_data: TEST_GRID,
  survey_points: TEST_SURVEY_POINTS,
}

export const TEST_SCENARIO_SUMMARY: ScenarioSummary = {
  name: 'Single Ferrous Target',
  file_name: 'single-ferrous-target',
  description: 'A single buried steel sphere for basic testing.',
  object_count: 1,
  terrain: {
    x_extent: [0.0, 20.0],
    y_extent: [0.0, 20.0],
    surface_elevation: 0.0,
  },
}

export const TEST_SCENARIO_DETAIL: ScenarioDetail = {
  ...TEST_SCENARIO_SUMMARY,
  objects: [
    {
      name: 'Steel sphere',
      object_type: 'ferrous',
      position: [10.0, 10.0, -1.0],
      radius: 0.05,
    },
  ],
  earth_field: [0.0, 20e-6, 45e-6],
  metadata: {},
  raw: {},
}
