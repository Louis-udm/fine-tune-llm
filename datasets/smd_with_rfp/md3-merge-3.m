Analyze the document contained by %%%% below. Within the document, locate the markdown table, then merge (combine) the original contents in all cells for each content row, and return in `JSON` format list. Note that the header of table is not the first content row. The definition of the `JSON` result is like:
```json
{{
  "request": "Merge original contents of each cell in every content row.",
  "rows": [
    "first row merged contents",
    "second row merged contents"
  ]
}}
```

Here is the document:

%%%%
 
 
 
 
ANNEX C- EVALUATION CRITERIA 
 
| ID   | Mandatory Requirement Description                                                                                                                                                                                                                                                                                                                               | Contractors Substantiation of Requirement   |
|------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------|
| M1   | The RPA solution must provide a bilingual (Official  languages– English and French) interface.                                                                                                                                                                                                                                                                  |                                             |
| M2.1 | The Contractor must provide a Robotic Process Automation  (RPA) solution that will operate on an enterprise-class  infrastructure using physical and virtual machines (VMs).                                                                                                                                                                                    |                                             |
| M2.2 | The RPA solution must include a design suite that will allow  ESDC to develop and deliver automated processes                                                                                                                                                                                                                                                   |                                             |
| M2.3 | The RPA solution must offer both attended and unattended  bot functions.                                                                                                                                                                                                                                                                                        |                                             |
| M2.4 | The RPA solution must have a bot administration suite that  allow ESDC to manage, schedule, and deploy automated  processes (for both attended and unattended automations)                                                                                                                                                                                      |                                             |
| M2.5 | The RPA solution must include autonomous agents,  meaning unattended bots that execute tasks and interact  with applications or systems independent of human  involvement.                                                                                                                                                                                      |                                             |
| M3   | The Contractor must provide a complete list of all  components and version numbers included in their solution  and must include:   a)  an Enterprise Optical character recognition (OCR)  engine.   b)  all database components required to support the  RPA solution  c)  all web browsers and versions it supports.  d)  All cloud base services it supports. |                                             |
| M4   | The solution must support hosting on a virtualized x86-64  architecture and must support hosting on a Windows 64-bit  operating systems running Windows Server 2016 and later  version.                                                                                                                                                                         |                                             |
| M5   | The solution must support the input and output of data  from the following file systems:   • Network File System (NFS)   • SAMBA                                                                                                                                                                                                                                |                                             |
| M6   | The solution must adhere to remote secure login through  ESDC network and remote desktop protocols.                                                                                                                                                                                                                                                             |                                             |
 
 
 
 
| M7   | The software must not require the use of Adobe Flash or Shockwave for any functionality.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | &#xfeff;   |
|------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| M8   | The solution must support Enterprise browser standards  without degradation in functionality.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |            |
| M9   | The solution must support the packaging and re- deployment of all required components within a given  automation process in order to migrate a process from one  RPA infrastructure environment to another such as from a  Development environment to Production environment.                                                                                                                                                                                                                                                                                                                                                                          |            |
| M10  | a The solution must integrate with directory services using  Lightweight Directory Access Protocol (LDAP) in order to:   ) Enforce role-based access control (RBAC) policies defined  by ESDC for both bots and authorized users;   b) Enforce authentication and authorization for any logical  access to information and solution resources; and   c) Enforce authentication and authorization before  performing any action that creates, views, updates,  transmits or deletes data.                                                                                                                                                               |            |
| M11  | For authentication credentials:   • the solution must obscure all authentication credentials  when entered into the RPA solution.   • the solution must obscure the display of all stored  authentication credentials once entered into the RPA  solution, including any logs.                                                                                                                                                                                                                                                                                                                                                                         |            |
| M12  | The solution must define, collect, and store audit records  and events associated with all user or bot operations listed  below:   a) successful and unsuccessful attempts to create, access,  modify, or delete security objects including audit data,  system configuration files and file or users' formal access  permissions;   b) successful and unsuccessful logon attempts;   c) privileged activities;   d) type of activity that occurred;   e) date and time the activity occurred;   f) where the activity occurred;   g) the source of activity;   h) success or failure outcome of activity; and   i) identity associated with activity. |            |
| M13  | The solution must employ cryptographic mechanisms for  end-to-end protection of data both when in motion or at  rest that have been approved by Communication Security  Establishment (CSE) and validated by the Cryptographic  Algorithm Validation Program (CAVP), and are specified in  the August 2016 ITSP.40.111 (https://www.cse- cst.gc.ca/en/system/files/pdf_documents/itsp.40.111- eng.pdf).                                                                                                                                                                                                                                                |            |
 
 
 
 
| ID   | Rated Requirement Description                                                                                                                                                                                                                           | Rating Criteria                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Max Points    |
|------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| R1.1 | The solution should provide written a  RPA strategic enterprise    implementation plan which will be  evaluated based on the following  marking scheme:   1.1 E- SDC infrastructure requirements;                                                       | 1 •  Information is well  documented / clear and  relevant to the subject – full  points per bullet  •  Information is semi  documented / unclear /  contains unrelated  information – half points per  bullet  .1 - ESDC infrastructure  requirements evaluation criteria   - includes relevant infrastructure  diagrams (Max 10 points)  - includes relevant infrastructure  hardware and software  requirements(Max 10 points)  - includes relevant costings and  firewall rule requirements(Max 10  points) | Max 30 points |
| R1.2 | The solution should provide written a  RPA strategic enterprise    implementation plan which will be  evaluated based on the following  marking scheme:   1.2 - best practices for business and IT  RPA Centres of Expertise (CoE);                     | •  Information is well  documented / clear and  relevant to the subject – full  points per bullet  •  Information is semi  documented / unclear /  contains unrelated  information – half points per  bullet    1.2 -best practices for business  and IT RPA centres of expertise  (CoE);   - includes relevant best practices  for business RPA center of  excellence. (Max 10 points)  - includes relevant best practices  for IT RPA center of excellence.  (Max 10 points)                                  | Max 20 points |
| R1.3 | 1 The solution should provide written a  RPA strategic enterprise    implementation plan which will be  evaluated based on the following  marking scheme:   .3 - business process evaluation and  creation of inventories of potential  RPA candidates; | •  Information is well  documented / clear and  relevant to the subject – full  points per bullet  •  Information is semi  documented / unclear /  contains unrelated  information – half points per  bullet                                                                                                                                                                                                                                                                                                    | Max 60 points |
 
 
 
 
| &#xfeff;   | &#xfeff;                                                                                                                                                                                                                                       | 1.3 - business process evaluation and creation of inventories of potential RPA candidates; (Max 25 points)) - includes relevant business process evaluation steps(Max 10 points) - includes relevant steps for the creation of inventories of candidates( Max 25 points)                                                                                                                                                                                                                                                                                                                                                                                            | &#xfeff;      |
|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| R1.4       | 1 The solution should provide written a  RPA strategic enterprise    implementation plan which will be  evaluated based on the following  marking scheme:   .4 - development, testing,  implementation, and maintenance of  the RPA solution ; | •  Information is well  documented / clear and  relevant to the subject – full  points per bullet  •  Information is semi  documented / unclear /  contains unrelated  information – half points per  bullet    1.4 - development, testing,  implementation, and maintenance  of the RPA solution ;  (Max 40  points)   - includes relevant information on  developing with the RPA solution  (Max 10 points)  - includes relevant information on  testing the RPA solution (Max 10  points)  - includes relevant information on  implementing the RPA solution  (Max 10 points)  - includes relevant information on  maintaining the RPA solution (Max  10 points) | Max 40 points |
| R1.5       | 1 The solution should provide written a  RPA strategic enterprise    implementation plan which will be  evaluated based on the following  marking scheme:   .5 - training;                                                                     | 1 •  Information is well  documented / clear and  relevant to the subject – full  points per bullet  •  Information is semi  documented / unclear /  contains unrelated  information – half points per  bullet  .5 - training; (Max 50 points)  -includes the training for strategy  for developers and administrators  (Max 10 points)                                                                                                                                                                                                                                                                                                                             | Max 50 points |

%%%%