from fpdf import FPDF
from datetime import datetime

class JalRakshakReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.set_text_color(30, 58, 138) # Dark Blue
        self.cell(0, 10, 'JALRAKSHAK - INDIA DRINKING WATER INTELLIGENCE', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_district_report(data):
    """
    Generates a PDF report for a specific district.
    data: dict containing district intelligence
    """
    pdf = JalRakshakReport()
    pdf.add_page()
    
    # District Identity
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, f"District Analysis: {data.get('name', 'Unknown')}", 0, 1)
    pdf.ln(5)
    
    # Section 1: Water Quality Snapshot
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, " 1. Water Quality Snapshot", 0, 1, fill=True)
    pdf.set_font('Arial', '', 11)
    wqi_val = data.get('wqi', data.get('WQI', None))
    wqi_display = f"{wqi_val:.2f}" if isinstance(wqi_val, (int, float)) else str(wqi_val or 'N/A')
    pdf.cell(0, 10, f"- Average Water Quality Index (WQI): {wqi_display}", 0, 1)
    
    # Calculate status using both keys
    raw_wqi = data.get('wqi', data.get('WQI', 0))
    status = "SAFE" if raw_wqi > 50 else "UNSAFE / CRITICAL"
    pdf.set_text_color(0, 128, 0) if status == "SAFE" else pdf.set_text_color(255, 0, 0)
    pdf.cell(0, 10, f"- Safety Classification: {status}", 0, 1)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    
    # Section 2: Infrastructure & Risk Factors
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, " 2. Infrastructure & Risk Factors", 0, 1, fill=True)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 10, f"- Population Density: {data.get('Pop_Density', 'N/A')} persons/sq.km", 0, 1)
    pdf.cell(0, 10, f"- Sewage Coverage: {round(data.get('Sewage_Coverage', 0), 2)}%", 0, 1)
    pdf.cell(0, 10, f"- Pipeline Age Index: {round(data.get('Pipe_Age_Index', 0), 2)}/50", 0, 1)
    pdf.ln(5)
    
    # Section 3: AI Intelligence Advisory
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, " 3. AI Intelligence Advisory (Claude Engine)", 0, 1, fill=True)
    pdf.set_font('Arial', 'I', 11)
    advisory = f"""Based on current data, {data.get('name')} is classified in the {data.get('Cluster', 'N/A')} risk cluster.
High population density combined with {status.lower()} water quality suggests immediate monitoring.
Recommended Action: Inspect pipeline integrity in high-density zones and issue a public health advisory."""
    pdf.multi_cell(0, 10, advisory)
    
    pdf.ln(20)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 10, "Authorized by JalRakshak National Intelligence Hub", 0, 1, 'C')
    
    return bytes(pdf.output())
