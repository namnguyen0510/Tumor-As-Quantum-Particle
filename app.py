import streamlit as st
from utils_options import *
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json


# Set page layout to wide
st.set_page_config(page_title = 'TRA CỨU THAM KHẢO KẾT QUẢ TUYỂN SINH NĂM 2024',layout="wide")
# Hide Streamlit's deploy button and three-dot menu
st.markdown("""
    <style>
        header {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Title
# Create columns for logo and title
col1, col2 = st.columns([1, 8])  # Adjust width ratio as needed

with col1:
    st.image("LogoNEU.png", width=256)  # Ensure "logo.png" exists in the same directory

with col2:
    st.title("TRA CỨU THAM KHẢO KẾT QUẢ TUYỂN SINH NĂM 2024")
tab3, tab2, tab5, tab4= st.tabs(['Quy đổi điểm', 'Bảng điểm tương đương năm 2025', 'Các câu hỏi thường gặp', 'Liên kết'])

with st.spinner('Đang tải dữ liệu và xử lý, vui lòng chờ...'):
    #neu_data_2022 = pd.read_csv('dset_private/neu-groups/neu-ts-gpa-2022.csv', sep = ';', low_memory = False)
    neu_data_2023 = pd.read_csv('dset_private/neu-groups/neu-ts-gpa-2023.csv', sep = ';', low_memory = False)
    neu_data_2024 = pd.read_csv('dset_private/neu-groups/neu-ts-gpa-2024.csv', sep = ';', low_memory = False)

    neu_data_2024 = neu_data_2024.drop(columns = ['MaSV', 'Họ đệm', 'Tên', 'Lớp',
        'Điểm TL10', 'Điểm TL4', 'MaHS', 'Số BD', 'Họ Tên',
        'Điểm ưu tiên', 'Khu vực ưu tiên', 'Đối tượng ưu tiên', 'Ngoại ngữ'
                                                ])
    neu_sat_2024 = neu_data_2024[neu_data_2024['Loại CC Nhóm 1'] == 'SAT']
    neu_tsa_2024 = neu_data_2024[neu_data_2024['Loại DGNL/DGTD Nhóm 2.1'] == 'DGNLDHBKHN']
    neu_hsa_2024 = neu_data_2024[neu_data_2024['Loại DGNL/DGTD Nhóm 2.1'] == 'DGNLDHQGHN']
    neu_apt_2024 = neu_data_2024[neu_data_2024['Loại DGNL/DGTD Nhóm 2.1'] == 'DGNLDHQGTPHCM']

    # Create combinations
    for col in ['Toán', 'Lý', 'Hóa', 'Văn', 'Tiếng Anh']:
        neu_data_2024[col] = (
            neu_data_2024[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )
    neu_data_2024['A00'] = neu_data_2024['Toán'] + neu_data_2024['Lý'] + neu_data_2024['Hóa']
    neu_data_2024['A01'] = neu_data_2024['Toán'] + neu_data_2024['Lý'] + neu_data_2024['Tiếng Anh']
    neu_data_2024['D01'] = neu_data_2024['Toán'] + neu_data_2024['Văn'] + neu_data_2024['Tiếng Anh']
    neu_data_2024['D07'] = neu_data_2024['Toán'] + neu_data_2024['Hóa'] + neu_data_2024['Tiếng Anh']

    neu_data_2023['A00'] = neu_data_2023['Toán'] + neu_data_2023['Lý'] + neu_data_2023['Hóa']
    neu_data_2023['A01'] = neu_data_2023['Toán'] + neu_data_2023['Lý'] + neu_data_2023['Tiếng Anh']
    neu_data_2023['D01'] = neu_data_2023['Toán'] + neu_data_2023['Văn'] + neu_data_2023['Tiếng Anh']
    neu_data_2023['D07'] = neu_data_2023['Toán'] + neu_data_2023['Hóa'] + neu_data_2023['Tiếng Anh']


    # Tab 1: Home
    thpt_sel = ['SAT', 'TSA', 'HSA', 'APT']
    with tab3:
        ga_selection = st.pills('**Chọn điểm ứng tuyển theo phương thức xét tuyển**', ga_options, selection_mode = 'single', default = ga_options[0])
        #st.write(ga_selection)
        if 'A00' in ga_selection:
            ref_df = pd.read_csv('dset_private/neu-groups/0_A00.csv')
            year_options = (2024, 2023, 2022, 'All')
        elif 'A01' in ga_selection:
            ref_df = pd.read_csv('dset_private/neu-groups/1_A01.csv')
            year_options = (2024, 2023, 2022, 'All')
        elif 'D01' in ga_selection:
            ref_df = pd.read_csv('dset_private/neu-groups/2_D01.csv')
            year_options = (2024, 2023, 2022, 'All')
        elif 'D07' in ga_selection:
            ref_df = pd.read_csv('dset_private/neu-groups/3_D07.csv')
            year_options = (2024, 2023, 2022, 'All')
        elif 'IA1' in ga_selection:
            ref_df = pd.read_csv('dset_private/neu-groups/4_IA1.csv')
            year_options = (2024, 2023, 2022, 'All')
        elif 'IA2' in ga_selection:
            ref_df = pd.read_csv('dset_private/neu-groups/5_IA2.csv')
            year_options = (2024, 2023, 2022, 'All')
        elif 'IA3' in ga_selection:
            ref_df = pd.read_csv('dset_private/neu-groups/6_IA3.csv')
            year_options = (2024, 2023, 2022, 'All')
        elif 'IA4' in ga_selection:
            ref_df = pd.read_csv('dset_private/neu-groups/7_IA4.csv')
            year_options = (2024, 2023, 2022, 'All')
        elif ga_selection in ['TSA','HSA','APT','SAT']:
            ref_df = pd.read_csv('dset_private/neu-groups/neu-2024-new.csv',low_memory=False)
            year_options = (2024, 2023, 2022, 'All')


        col1, col2 = st.columns([1,4])
        with col1:
            with st.container(border = True):
                st.write('**Nhập điểm của bạn**')
                if ga_selection not in thpt_sel:
                    ga_comp = ga_selection.split('(')[1].split(')')[0].split(',')
                    ga_comp = [x.strip() for x in ga_comp]
                    grade_01 = st.number_input(f"Điểm {ga_comp[0]}")
                    grade_02 = st.number_input(f"Điểm {ga_comp[1]}")
                    grade_03 = st.number_input(f"Điểm {ga_comp[2]}")
                    user_score = grade_01+grade_02+grade_03
                    st.write("Tổng điểm", np.round(grade_01+grade_02+grade_03,2))
                elif ga_selection == 'SAT':
                    user_score = st.number_input(f"Điểm {ga_selection}", min_value=0, max_value=1600, step=10, format="%d")
                else:
                    user_score = st.number_input(f"Điểm {ga_selection}")
                    st.write("Tổng điểm", user_score)
            #if ga_selection not in thpt_sel:
                year_selection = st.selectbox('**Chọn Năm so sánh**', year_options, index=0)
            #else:
                #year_selection = 2024
        with col2:
            if ga_selection not in thpt_sel:
                if year_selection != 'All':
                    ref_df = ref_df[ref_df['Năm'] == year_selection]
                user_score = grade_01+grade_02+grade_03
                bench_score = ref_df[ga_comp[0]]+ref_df[ga_comp[1]]+ref_df[ga_comp[2]]
            elif 'SAT' in ga_selection:
                ref_df = ref_df[ref_df['TT Nhóm'] == 'SAT_ACT']
                ref_df = ref_df[ref_df['Điểm ưu tiên'] == '0']
                ref_df = ref_df[ref_df['Loại CC Nhóm 1'] == 'SAT']
                bench_score = ref_df['Điểm CCQT Nhóm 1'].astype(str).str.replace(",", ".").astype(float)
                #print(min(bench_score),max(bench_score))
                #print(ref_df['TT Tên ngành'])
            #  'TSA', 'HSA', 'APT'
            elif ga_selection == 'TSA':
                ref_df = ref_df[ref_df['TT Nhóm'] == 'NHOM2']
                ref_df = ref_df[ref_df['Điểm ưu tiên'] == '0']
                ref_df = ref_df[ref_df['Loại DGNL/DGTD Nhóm 2.1'] == 'DGNLDHBKHN']
                bench_score = ref_df['Điểm DGNL/DGTD 2.1'].astype(str).str.replace(",", ".").astype(float)
                #print(min(bench_score),max(bench_score))
                #print(ref_df['TT Tên ngành'])
            elif ga_selection == 'HSA':
                ref_df = ref_df[ref_df['TT Nhóm'] == 'NHOM2']
                ref_df = ref_df[ref_df['Điểm ưu tiên'] == '0']
                ref_df = ref_df[ref_df['Loại DGNL/DGTD Nhóm 2.1'] == 'DGNLDHQGHN']
                bench_score = ref_df['Điểm DGNL/DGTD 2.1'].astype(str).str.replace(",", ".").astype(float)
                #print(min(bench_score),max(bench_score))
                #print(ref_df['TT Tên ngành'])
            elif ga_selection == 'APT':
                ref_df = ref_df[ref_df['TT Nhóm'] == 'NHOM2']
                ref_df = ref_df[ref_df['Điểm ưu tiên'] == '0']
                ref_df = ref_df[ref_df['Loại DGNL/DGTD Nhóm 2.1'] == 'DGNLDHQGTPHCM']
                bench_score = ref_df['Điểm DGNL/DGTD 2.1'].astype(str).str.replace(",", ".").astype(float)
                #print(min(bench_score),max(bench_score))
                #print(ref_df['TT Tên ngành'])

            if ga_selection not in thpt_sel:
                if compute_position(user_score,bench_score)[0] <= len(ref_df):
                    with st.container(border=True):
                        st.markdown(f'''
                        Số điểm của bạn là: **{np.round(user_score,2)}**. So sánh với tổng số hồ sơ ứng tuyển năm **{year_selection}**, điểm của bạn ở thứ tự **{compute_position(user_score,bench_score)[0]}/{len(ref_df)}** hồ sơ.\n
                        Xếp hạng điểm **{ga_comp[0]}**: **{compute_position(grade_01,ref_df[ga_comp[0]])[0]}**/{len(ref_df)}.
                        Xếp hạng điểm **{ga_comp[1]}**: **{compute_position(grade_02,ref_df[ga_comp[1]])[0]}**/{len(ref_df)}.
                        Xếp hạng điểm **{ga_comp[2]}**: **{compute_position(grade_03,ref_df[ga_comp[2]])[0]}**/{len(ref_df)}.
                        ''')
                    if user_score != 0:
                        MAJOR = []
                        POSITION = []
                        PERCENTILE = []

                        # Select data based on the year
                        year_data_map = {2024: neu_data_2024, 2023: neu_data_2023}
                        data = year_data_map.get(year_selection, None)

                        if data is not None:
                            # Define a dictionary to map the GA selections to the corresponding column
                            ga_column_map = {
                                'A00 (Toán, Lý, Hóa)': 'A00',
                                'A01 (Toán, Lý, Tiếng Anh)': 'A01',
                                'D01 (Toán, Văn, Tiếng Anh)': 'D01',
                                'D07 (Toán, Hóa, Tiếng Anh)': 'D07'
                            }
                            
                            # Get the correct column based on GA selection
                            ga_column = ga_column_map.get(ga_selection, None)

                            if ga_column:
                                for major in data['Ngành'].unique():
                                    ref_grade = data[data['Ngành'] == major][ga_column]

                                    # Compute user position and percentile only once
                                    user_position, user_percentile = compute_position(user_score, ref_grade)
                                    user_percentile = np.round(user_percentile, 2)
                                    user_position = np.round(user_position, 2)

                                    if 0 < user_percentile <= 100:
                                        MAJOR.append(major)
                                        PERCENTILE.append(user_percentile)
                                        POSITION.append(user_position)

                            # Create the dataframe and sort the values
                            df = pd.DataFrame({'Ngành': MAJOR, 'Percentile': PERCENTILE, 'Position': POSITION})
                            
                            # Group by 'Ngành' and calculate the average 'Percentile' for each major
                            df_sorted = df.groupby('Ngành', as_index=False)['Percentile'].mean()

                            # Sort by 'Percentile' and reset index
                            df_sorted = df_sorted.sort_values(by='Percentile', ascending=True).reset_index(drop=True)

                            # Set the rank as index
                            df_sorted.index = range(1, len(df_sorted) + 1)

                            half = len(df_sorted) // 2
                        with st.expander(f'**Chi tiết *({len(df_sorted)})* ngành/chương trình trúng tuyển (năm {year_selection})**'):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.dataframe(df_sorted['Ngành'].iloc[:half], use_container_width=True)

                            with col2:
                                st.dataframe(df_sorted['Ngành'].iloc[half:], use_container_width=True)

                        with st.expander(f'**Phân tích xét tuyển theo *({len(df_sorted)})* ngành/chương trình tại NEU (năm {year_selection})**'):
                            # Create the bar plot using plotly
                            fig = go.Figure(go.Bar(
                                y=df_sorted['Ngành'],  # Sorted majors
                                x=df_sorted['Percentile'],  # Sorted positions
                                orientation='h',
                                marker=dict(color='skyblue'),
                                text=df_sorted['Percentile'],  # Text inside the bar
                                    textposition='inside',         # Place it inside the bar
                                    textangle=0,  # Ensure text is horizontal (angle = 0)
                                    textfont=dict(
                                        size=44,                  # Increase font size
                                        color='black',              # Text color (black on light bar)
                                    )
                            ))

                            # Update layout for better aesthetics
                            fig.update_layout(
                                #title='Ranking Majors by Position',
                                xaxis_title='Top %',
                                yaxis_title='Ngành',
                                height=1200,
                                width=800,
                                yaxis=dict(autorange='reversed')  # Key: shows best ranks at the top
                            )
                            st.plotly_chart(fig)
                else:
                    st.markdown(f'''
                        Số điểm {ga_selection} của bạn là {user_score} **không trúng tuyển** vào NEU năm {year_selection}. Điểm tối thiểu {ga_selection} là **{min(bench_score)}**.             
                        ''')
                
            else:
                if user_score != 0:
                    MAJOR = []
                    POSITION = []
                    PERCENTILE = []
                    NUM_CANDIDATES = []  # New variable for number of entries per major

                    # Define the correct score column based on exam type
                    if ga_selection == 'SAT':
                        score_column = 'Điểm CCQT Nhóm 1'
                    elif ga_selection in ['TSA', 'HSA', 'APT']:
                        score_column = 'Điểm DGNL/DGTD 2.1'
                    else:
                        score_column = None

                    if score_column:
                        for major in ref_df['TT Tên ngành']:
                            ref_grade = ref_df[ref_df['TT Tên ngành'] == major][score_column].astype(str).str.replace(",", ".").astype(float)
                            if len(ref_grade) > 0:
                                user_position, user_percentile = compute_position(user_score, ref_grade)
                                user_percentile = np.round(user_percentile, 2)
                                user_position = int(user_position)
                                if 0 < user_percentile and user_percentile<=100:
                                    MAJOR.append(major)
                                    PERCENTILE.append(user_percentile)
                                    POSITION.append(user_position)
                                    NUM_CANDIDATES.append(len(ref_grade))  # Save number of entries

                    # Create DataFrame
                    df = pd.DataFrame({
                        'Ngành': MAJOR,
                        'Percentile': PERCENTILE,
                        'Position': POSITION,
                        'Num_Candidates': NUM_CANDIDATES  # New column
                    })
                    # Sort and group
                    df_sorted = df.sort_values(by='Percentile', ascending=True)
                    df_sorted = df_sorted.groupby('Ngành', as_index=False).agg({
                        'Percentile': 'mean',
                        'Position': 'mean',
                        'Num_Candidates': 'mean'  # Average or you can use 'max' if you prefer
                    })

                    df_sorted['Percentile'] = df_sorted['Percentile'].astype(float)
                    df_sorted['Position'] = df_sorted['Position'].astype(int)
                    df_sorted['Num_Candidates'] = df_sorted['Num_Candidates'].astype(int)  # Ensure integer type

                    df_sorted = df_sorted.sort_values(by='Percentile', ascending=True).reset_index(drop=True)
                    df_sorted.index = range(1, len(df_sorted) + 1)

                    half = len(df_sorted) // 2
                    if year_selection == 2024:
                        if compute_position(user_score,bench_score)[0] > len(bench_score):
                            st.markdown(f'''
                            Số điểm {ga_selection} của bạn là {user_score} **không trúng tuyển** vào NEU năm 2024. Điểm tối thiểu {ga_selection} là **{min(bench_score)}**.             
                            ''')
                        else:
                            with st.container(border=True):
                                st.markdown(f'''
                                So sánh với tổng số hồ sơ ứng tuyển năm **{year_selection}**, bạn đứng thứ **{compute_position(user_score,bench_score)[0]}/{len(bench_score)}** thí sinh ứng tuyển với điểm {ga_selection} và NEU.
                                ''')

                            
                                with st.expander(f'**Chi tiết *({len(df_sorted)})* ngành/chương trình trúng tuyển (năm {year_selection})**'):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.dataframe(df_sorted['Ngành'].iloc[:half], use_container_width=True)

                                    with col2:
                                        st.dataframe(df_sorted['Ngành'].iloc[half:], use_container_width=True)

                                with st.expander(f'**Phân tích xét tuyển theo *({len(df_sorted)})* ngành/chương trình tại NEU (năm {year_selection})**'):
                                    # Create a label: "Position/Total" for text on bar
                                    df_sorted['Text_Label'] = df_sorted['Position'].astype(str) + '/' + df_sorted['Num_Candidates'].astype(str)

                                    # Create the bar plot using plotly
                                    fig = go.Figure(go.Bar(
                                        y=df_sorted['Ngành'],  # Only major name
                                        x=df_sorted['Percentile'],  # Percentile score
                                        orientation='h',
                                        marker=dict(color='skyblue'),
                                        text=df_sorted['Text_Label'],  # Text inside the bar
                                        textposition='inside',         # Place it inside the bar
                                        textangle=0,  # Ensure text is horizontal (angle = 0)
                                        textfont=dict(
                                            size=44,                  # Increase font size
                                            color='black',              # Text color (black on light bar)
                                        )
                                    ))

                                    # Update layout for better aesthetics
                                    fig.update_layout(
                                        xaxis_title='Top %',
                                        yaxis_title='Ngành',
                                        height=1200,
                                        width=800,
                                        yaxis=dict(autorange='reversed')  # Best ranks at the top
                                    )
                                    st.plotly_chart(fig)
                                    ex_per =  df_sorted['Percentile'].iloc[0]
                                    ex_maj =  df_sorted['Ngành'].iloc[0]
                                    ex_pos = df_sorted['Text_Label'].iloc[0]
                                    with st.container(border = True):
                                        st.markdown(f'Ví dụ: Điểm ứng tuyển bài thi **{ga_selection}** của bạn đứng **top-{ex_per}**% ({ex_pos} hồ sơ) của ngành {ex_maj} tại NEU.')
                    else:
                        st.markdown('Dữ liệu đang cập nhật...')

with tab2:
    with st.spinner("Đang tải dữ liệu tuyển sinh 2024..."):
        #df = pd.read_csv('dset_private/reference_grade_df_v1.csv', sep = ';', low_memory=False)
        #df.columns = ['Điểm TN THPT', 'HSA', 'TSA', 'APT', 'SAT']
        #st.dataframe(df)
        st.markdown('Đại học sẽ công bố theo kế hoạch chung của Bộ GD&ĐT, thời gian công bố muộn nhất cùng thời gian công bố ngưỡng bảo đảm chất lượng đầu vào.')
with tab4:
    st.markdown('**Trang chủ Đại học Kinh tế Quốc dân:** https://neu.edu.vn/')
    st.markdown('**Cửa sổ tư vấn tuyển sinh tương tác trực tuyến ORLABNEU**: : https://daotao.neu.edu.vn')
    st.markdown('**Các ngành/chương trình đào tạo ĐHCQ năm 2025:** https://courses.neu.edu.vn/')
    st.markdown('**ChatBot hỗ trợ sinh viên và thí sinh của Trường Công Nghệ:** https://neutech.ai.vn/')

# Load the faqs.json file
with open('faqs.json', 'r', encoding='utf-8') as f:
    faqs = json.load(f)
# Inside Tab 5 (the FAQ tab)
with tab5:
    st.header("Câu hỏi thường gặp")
    # Loop through FAQs
    for i in range(1, (len(faqs) // 2) + 1):
        question = faqs.get(f"Q{i}", "")
        answer = faqs.get(f"A{i}", "")
        if question and answer:
            with st.expander(question):
                st.markdown(answer)

import streamlit.components.v1 as components

# Inject Google Analytics
components.html(
    """
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-PB1EE17KP3"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-PB1EE17KP3');
    </script>
    """,
    height=0,  # No visual space taken
)