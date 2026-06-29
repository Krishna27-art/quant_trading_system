from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from utils.structured_logger import get_structured_logger

logger = get_structured_logger("tracing")

# Initialize OpenTelemetry Tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer("quant_system")

try:
    # Export to local Jaeger instance running on port 4317
    otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
    span_processor = BatchSpanProcessor(otlp_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    logger.info("OpenTelemetry tracing processor configured to export to Jaeger (localhost:4317).")
except Exception as e:
    logger.warning(
        f"Could not connect OpenTelemetry to Jaeger ({e}). Tracing will run locally in-memory."
    )
